"""
[M1] 规则评分引擎

职责：不调用任何大模型，纯靠代码对每条应答打出 0–5 分的「规则分」。

设计原则（这条很重要，容易被问）：
    规则只检测**应答的行为特征**，不匹配「未回答」「未给出」这类结论词。
    原因：原始数据的 answer 字段是人工撰写的摘要，摘要里本来就写了人的判断。
    如果规则去匹配这些词，等于拿人工的结论反推人工的标签 —— 那验证就失去意义了。
    所以这里只认五类可独立观测的行为：导流、缺路径、缺口径、模板复用、无实质内容。

运行方式（项目根目录）：
    python -m src.rules
"""

import json
import re
import sys
from difflib import SequenceMatcher

from src.config import MAX_SCORE, OUTPUT_DIR, PASS_THRESHOLD, RECORDS_JSON


# ============================================================
# 一、判据（正则）
# ============================================================

# --- 规则 1：导流甩锅 ---
# 用户问「能不能 / 多久」，AI 却把问题推给「查订单 / 看记录」等操作环节。
REDIRECT_RE = re.compile(
    r"让核对|选一笔|请提供|导流|转为查询|转向查询|转而|查询具体|查询该账户"
)

# --- 规则 2：缺时间口径 ---
# 用户问了时效类问题，回答里却没有任何具体时间词。
QUESTION_TIME_RE = re.compile(r"多久|什么时候|要多久|多长时间|几天|何时")
ANSWER_TIME_RE = re.compile(
    r"实时|立即|当日|当天|工作日|分钟|小时|T\+1|个月|周期|定期|每月|\d+\s*天|\d+\s*小时"
)

# --- 规则 3：缺操作路径 ---
# 用户问「怎么 / 在哪里」，回答里却给不出可照着点的路径。
QUESTION_PATH_RE = re.compile(r"怎么|如何|在哪里|在哪|怎样")
ANSWER_PATH_RE = re.compile(r"→|->|入口|路径|步骤")

# --- 规则 5：无实质内容 ---
NO_SUBSTANCE_RE = re.compile(r"未作答|直接弹出人工")

# --- 预处理：剔除「未给…」这类否定分句 ---
# 作用：避免「未给进度查询入口」里的「入口」被误当成「给了入口」。
# 注意只剔除「未+给/提供/列出/回答」这种明确的否定分句，
# 像「未直接承诺时限」这种不剔除 —— 因为它本身就是正常答案的一部分。
NEGATION_RE = re.compile(r"未(?:给|提供|列出|回答)[^，。；]*")


def strip_negation(answer: str) -> str:
    """把否定的分句去掉，剩下的部分才用来判断「有没有给路径 / 有没有给时间」。"""
    return NEGATION_RE.sub("", answer)


# ============================================================
# 二、检测器
# 每个检测器接收 (当前记录, 上一条记录)，命中则返回一句人话说明，否则返回 None。
# ============================================================


def detect_redirect(record: dict, prev: dict | None) -> str | None:
    """R1 导流甩锅：把规则型问题当成个人数据查询，没给结论就把人支走。"""
    m = REDIRECT_RE.search(record["answer"])
    if m:
        return f"命中「{m.group()}」：未直接给结论，把问题推到查询/操作环节"
    return None


def detect_missing_time(record: dict, prev: dict | None) -> str | None:
    """R2 缺时间口径：问了时效，答里没有具体时间。"""
    if not QUESTION_TIME_RE.search(record["question"]):
        return None
    if ANSWER_TIME_RE.search(strip_negation(record["answer"])):
        return None
    return "问的是时效，但回答里没有任何具体时间口径"


def detect_missing_path(record: dict, prev: dict | None) -> str | None:
    """R3 缺操作路径：问了怎么操作，答里没有可照着点的路径。"""
    if not QUESTION_PATH_RE.search(record["question"]):
        return None
    if ANSWER_PATH_RE.search(strip_negation(record["answer"])):
        return None
    return "问的是操作方法，但回答里没有可照做的路径或入口"


def detect_template_reuse(record: dict, prev: dict | None) -> str | None:
    """R4 模板复用：相邻两轮回答高度雷同，说明走的是一套固定模板。

    说明：本版用 difflib 算文本相似度。由于手头数据的 answer 是「摘要」而不是
    原始应答全文，这条规则在摘要上基本抓不到东西 —— 这是个已知局限，
    也正好说明「纯规则」不够用，需要大模型做语义层判断。
    """
    if prev is None:
        return None
    ratio = SequenceMatcher(None, prev["answer"], record["answer"]).ratio()
    if ratio > 0.8:
        return f"与上一轮（第 {prev['id']} 条）回答相似度 {ratio:.0%}，疑似复用同一模板"
    return None


def detect_no_substance(record: dict, prev: dict | None) -> str | None:
    """R5 无实质内容：没作答，或者直接甩人工入口。"""
    m = NO_SUBSTANCE_RE.search(record["answer"])
    if m:
        return f"命中「{m.group()}」：没有给出实质答复"
    return None


# ============================================================
# 三、规则清单
# 四元组：规则代号、规则名、命中扣分、检测函数
# 扣分权重的依据：越接近「完全没回答用户的问题」，扣得越多。
# ============================================================

RULE_SPECS = [
    ("R1", "导流甩锅", 2.6, detect_redirect),
    ("R2", "缺时间口径", 2.2, detect_missing_time),
    ("R3", "缺操作路径", 1.8, detect_missing_path),
    ("R4", "模板复用", 2.4, detect_template_reuse),
    ("R5", "无实质内容", 2.6, detect_no_substance),
]


# ============================================================
# 四、评分
# ============================================================


def score_one(record: dict, prev: dict | None = None) -> dict:
    """给单条记录打规则分。返回 {id, rule_score, hits}。"""
    hits = []
    for code, name, deduct, fn in RULE_SPECS:
        detail = fn(record, prev)
        if detail:
            hits.append({"rule": f"{code} {name}", "deduct": deduct, "detail": detail})

    total_deduct = sum(h["deduct"] for h in hits)
    score = round(max(0.0, MAX_SCORE - total_deduct), 1)

    return {
        "id": record["id"],
        "rule_score": score,
        "rule_label": "答对" if score >= PASS_THRESHOLD else "答偏",
        "hits": hits,
    }


def score_all(records: list[dict]) -> list[dict]:
    """按顺序给全部记录打分。prev 指的是列表里上一条。"""
    results = []
    for i, record in enumerate(records):
        prev = records[i - 1] if i > 0 else None
        results.append(score_one(record, prev))
    return results


def compare_with_human(records: list[dict], results: list[dict]) -> dict:
    """把规则判定和人工标注逐条对照。

    这一段其实就是 [M3] 一致性验证的雏形 —— 先算出「规则有多准」，
    后面 DAY 4 会把它扩展成含混淆矩阵和 Kappa 的完整版本。
    """
    by_id = {r["id"]: r for r in results}
    miss, false_alarm = [], []

    for rec in records:
        human = rec["human_label"]
        rule = by_id[rec["id"]]["rule_label"]
        if human == rule:
            continue
        if human == "答偏" and rule == "答对":
            miss.append(rec["id"])          # 漏检：人认为错，规则放过
        else:
            false_alarm.append(rec["id"])   # 误伤：人认为对，规则判错

    total = len(records)
    correct = total - len(miss) - len(false_alarm)

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
        "miss": miss,
        "false_alarm": false_alarm,
    }


# ============================================================
# 五、入口
# ============================================================


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 58)
    print("[M1] 规则评分引擎")
    print("=" * 58)

    print(f"\n[1/2] 加载 {RECORDS_JSON.name} ...")
    records = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    print(f"      {len(records)} 条")

    print("\n[2/2] 逐条评分 ...\n")
    results = score_all(records)

    print(f"  {'序号':<5}{'规则分':<8}{'规则判定':<10}命中规则")
    print("  " + "-" * 56)
    for rec_r in results:
        hit_text = " · ".join(h["rule"] for h in rec_r["hits"]) or "—"
        print(f"  {rec_r['id']:<6}{rec_r['rule_score']:<9}{rec_r['rule_label']:<11}{hit_text}")

    cmp = compare_with_human(records, results)

    print("\n" + "-" * 58)
    print("规则判定 vs 人工标注")
    print(f"  总条数        {cmp['total']}")
    print(f"  一致          {cmp['correct']}")
    print(f"  一致率        {cmp['accuracy']:.1%}")
    print(f"  漏检（人判偏、规则放过）  {cmp['miss'] or '无'}")
    print(f"  误伤（人判对、规则判偏）  {cmp['false_alarm'] or '无'}")
    print("-" * 58)

    output_path = OUTPUT_DIR / "rule_results.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已写出：{output_path.name}")
    print("\n下一步：DAY 3 接入大模型评审 src/llm_judge.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
