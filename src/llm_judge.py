"""
[M2] 大模型评审（LLM-as-Judge）

职责：把每条问答发给大模型，按 src/prompts.py 里的评分量表打分，输出结构化结果。

这一段代码处理了三个真实项目必踩的坑：
    坑 1 · 模型输出不干净。它会把 JSON 包在 ```json 代码块里，或者前后加一句
           「好的，我来分析一下」——所以必须写容错解析，不能直接 json.loads。
    坑 2 · 网络会抖动、接口会限流。必须重试，而且重试之间要等一下。
    坑 3 · 模型偶尔不守规矩，字段缺失或类型不对（比如把 3 写成 "3"）。
           必须校验 + 转换，不能想当然。

运行方式（项目根目录）：
    python -m src.llm_judge              # 跑全部 30 条
    python -m src.llm_judge --limit 3    # 先拿 3 条试水（推荐第一次这么跑）
"""

import json
import re
import sys
import time

import requests

from src.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_RETRY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OUTPUT_DIR,
    RECORDS_JSON,
)
from src.prompts import build_messages

# 每条评审之间歇一下，避免请求太密被接口限流
REQUEST_INTERVAL = 0.3


# ============================================================
# 一、调用
# ============================================================


def call_llm(messages: list[dict]) -> str:
    """发一次请求，返回模型输出的原始文本。

    DeepSeek 的接口格式和 OpenAI 一致：
        POST {base_url}/chat/completions
        Header: Authorization: Bearer <key>
        Body:   {model, messages, temperature, response_format}
    """
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        # 温度 0：要的是每次都一样的判定，不要「创造力」
        "temperature": LLM_TEMPERATURE,
        # 要求接口层面就返回 JSON 对象（DeepSeek 支持这个参数）
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ============================================================
# 二、容错解析（坑 1 的解法）
# ============================================================


def extract_json(text: str) -> dict:
    """从模型输出里把 JSON 抠出来。

    处理三种情况：
        a. 干净的 JSON                → 直接解析
        b. ```json { ... } ``` 包裹的  → 先剥代码块
        c. 前后带解释文字的            → 取第一个 { 到最后一个 }
    """
    # 情况 b：剥掉 markdown 代码围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)

    # 情况 c：从第一个左花括号截到最后一个右花括号
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"输出里找不到 JSON：{text[:120]}")

    return json.loads(text[start : end + 1])


def normalize_result(raw: dict) -> dict:
    """校验并规整模型返回的结果（坑 3 的解法）。

    模型可能把 score 返成 "3" 这种字符串，也可能漏字段 —— 都在这里兜住。
    """
    if "score" not in raw:
        raise ValueError(f"结果缺 score 字段：{raw}")

    score = int(float(raw["score"]))          # "3" / 3.0 / 3 都能吃下
    score = max(0, min(5, score))             # 越界就夹回 0–5

    return {
        "score": score,
        "reason_tag": str(raw.get("reason_tag", "不适用")).strip(),
        "reason": str(raw.get("reason", "")).strip()[:60],
    }


# ============================================================
# 三、单条评审（带重试）
# ============================================================


def judge_one(record: dict) -> dict:
    """评一条记录。失败就重试，重试到上限还没成则抛出异常。"""
    messages = build_messages(record["question"], record["answer"])

    last_err = None
    for attempt in range(1, LLM_MAX_RETRY + 1):
        try:
            text = call_llm(messages)
            return normalize_result(extract_json(text))
        except Exception as e:                # noqa: BLE001  网络、解析、字段问题都在这兜
            last_err = e
            if attempt < LLM_MAX_RETRY:
                time.sleep(1.5 * attempt)     # 退避：第 1 次等 1.5s，第 2 次等 3s

    raise RuntimeError(f"重试 {LLM_MAX_RETRY} 次仍失败：{last_err}")


# ============================================================
# 四、入口
# ============================================================


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 支持 --limit N，方便先小批量试跑
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    print("=" * 58)
    print("[M2] 大模型评审")
    print("=" * 58)

    # 没配 Key 就友好提示，而不是让它后面抛一个看不懂的 401
    if not LLM_API_KEY:
        print("\n  [X] 没有读到 LLM_API_KEY")
        print("      请复制 .env.example 为 .env，把 DeepSeek 的 Key 填进去：")
        print("      1. 在项目根目录新建文件 .env（注意前面有那个点）")
        print("      2. 写入一行：LLM_API_KEY=sk-你的key")
        print("      3. 重新运行本命令")
        return 1

    print(f"\n模型：{LLM_MODEL}")
    print(f"接口：{LLM_BASE_URL}")

    records = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    if limit:
        records = records[:limit]
    print(f"待评：{len(records)} 条" + ("（--limit 模式）" if limit else ""))

    print("\n开始逐条评审 ...\n")

    results, failed = [], []
    for i, rec in enumerate(records, 1):
        try:
            r = judge_one(rec)
            results.append({"id": rec["id"], **r})
            flag = "答对" if r["score"] >= 4 else "答偏"
            print(f"  [{i:>2}/{len(records)}] #{rec['id']:<3} {r['score']} 分 {flag}"
                  f"  {r['reason_tag']}  {r['reason']}")
        except Exception as e:                # noqa: BLE001
            failed.append(rec["id"])
            print(f"  [{i:>2}/{len(records)}] #{rec['id']:<3} 失败：{e}")

        time.sleep(REQUEST_INTERVAL)

    print("\n" + "-" * 58)
    print(f"成功 {len(results)} 条，失败 {len(failed)} 条" + (f"（{failed}）" if failed else ""))
    print("-" * 58)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / ("llm_results_partial.json" if limit else "llm_results.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写出：{out.name}")

    if not limit:
        print("\n下一步：DAY 4 写 src/analyze.py —— 把规则分和 LLM 分合并，")
        print("        再和你的人工标注比对，算一致率和 Cohen's Kappa。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
