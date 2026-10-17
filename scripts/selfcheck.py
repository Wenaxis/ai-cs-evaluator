"""
自检：不联网、不花一分钱，验证「改动到底有没有生效」。

为什么需要这个东西：

    跑完 30 条评测，数字好看，但那只能说明「这一次跑得好」。
    面试官问「你怎么证明是你改的东西起作用，不是碰巧」，光看分数答不了。

    这份自检回答的是更早的一个问题：**代码真的是我以为的那样吗？**
    —— 知识库文件在不在、检索能不能命中、工具注册了没、prompt 准则还在不在、
      exclude_id 是不是还强制传。

    这些都不用调模型，几秒钟跑完。它们全过，才值得花那几毛钱去跑真评测。
    任何一项挂了，先修代码，别浪费 API 额度。

用法（项目根目录）：

    .venv/Scripts/python.exe scripts/selfcheck.py            # 只报汇总
    .venv/Scripts/python.exe scripts/selfcheck.py --table    # 外加 30 条检索对照表

退出码：全过 0，有失败 1（可以直接接进 CI）。
"""

from __future__ import annotations

import inspect
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

results: list[tuple[bool, str, str]] = []   # (通过?, 项名, 说明)


def check(ok: bool, title: str, detail: str = "") -> None:
    results.append((ok, title, detail))
    mark = "OK  " if ok else "FAIL"
    print(f"      {mark} {title}" + (f"：{detail}" if detail else ""))


def section(no: int, total: int, name: str) -> None:
    print(f"\n[{no}/{total}] {name}")


# ============================================================
# 1. 知识库能否加载
# ============================================================


def check_knowledge_file() -> None:
    from src.agent.knowledge import KnowledgeBase
    from src.config import KNOWLEDGE_JSON

    section(1, 7, "知识库文件")

    if not KNOWLEDGE_JSON.exists():
        check(False, "cs_rules.json 存在", f"找不到 {KNOWLEDGE_JSON}")
        return
    check(True, "cs_rules.json 存在")

    try:
        kb = KnowledgeBase()
    except Exception as exc:                       # noqa: BLE001
        check(False, "能构造 KnowledgeBase", f"{type(exc).__name__}: {exc}")
        return
    check(True, "能构造 KnowledgeBase")

    n = len(kb.rules)
    check(n > 0, "规则条数 > 0", f"{n} 条")

    src = Counter(r.get("source", "?") for r in kb.rules)
    check(
        True,
        "来源分布",
        " / ".join(f"{k} {v}" for k, v in sorted(src.items())),
    )

    scenes = Counter(r.get("scene", "?") for r in kb.rules)
    check(True, "场景分布", " / ".join(f"{k} {v}" for k, v in sorted(scenes.items())))


# ============================================================
# 2. 用评测集的 30 个问题去查，看能不能查到东西
# ============================================================


def check_retrieval(table: bool = False) -> list[tuple[float, int, str, str]]:
    from src.agent.knowledge import KnowledgeBase
    from src.config import RECORDS_JSON

    section(2, 7, "知识库检索覆盖")

    recs = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    recs = recs if isinstance(recs, list) else recs["records"]
    kb = KnowledgeBase()

    rows: list[tuple[float, int, str, str]] = []
    for r in recs:
        hits = kb.search(r["question"], k=1)
        score = hits[0]["score"] if hits else 0.0
        topic = hits[0]["topic"] if hits else "（无命中）"
        rows.append((score, r["id"], r["question"], topic))

    zero = [r for r in rows if r[0] <= 0]
    check(
        not zero,
        f"{len(rows)} 条问题都能检索到规则",
        f"零命中 {len(zero)} 条" if zero else f"命中率 {len(rows) - len(zero)}/{len(rows)}",
    )

    if zero:
        for _, cid, q, _ in zero:
            print(f"           #{cid} {q}")

    best = max(rows, key=lambda r: r[0])
    worst = min(rows, key=lambda r: r[0])
    check(
        True,
        "得分区间",
        f"{worst[0]:.2f} ~ {best[0]:.2f}（低分那条是 #{worst[1]}「{worst[2]}」→ {worst[3]}）",
    )

    if table:
        print("\n      ── 30 条检索对照表（人眼扫一遍，比任何指标都可靠）──")
        for score, cid, q, topic in sorted(rows, key=lambda r: r[0]):
            print(f"      {score:6.2f}  #{cid:<3} {q[:22]:<24} → {topic}")

    return rows


# ============================================================
# 3. 工具注册与默认装配
# ============================================================


def check_tools() -> None:
    from src.agent.tools import TOOL_SCHEMAS, ToolRuntime

    section(3, 7, "工具注册")

    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    check("search_knowledge" in names, "search_knowledge 已注册")
    check("search_precedents" in names, "search_precedents 已注册")

    if names and names[0] == "search_knowledge":
        check(True, "search_knowledge 排在工具表第一位", "模型看工具是按顺序看的")
    else:
        check(False, "search_knowledge 排在工具表第一位", f"实际第一位是 {names[0] if names else '空'}")

    # 不传 kb 时，ToolRuntime 应该自己去加载，而不是留一个 None 让后面炸
    sig = inspect.signature(ToolRuntime.__init__)
    check(
        "kb" in sig.parameters and sig.parameters["kb"].default is None,
        "ToolRuntime 接受 kb=None 并自动加载",
    )


# ============================================================
# 4. prompt 准则还在不在
# ============================================================


def check_prompt() -> None:
    from src.agent.cs_agent import SYSTEM_PROMPT

    section(4, 7, "prompt 准则")

    check(
        "search_knowledge" in SYSTEM_PROMPT and "知识库" in SYSTEM_PROMPT,
        "有「先查知识库」的指引",
    )
    check("转人工" in SYSTEM_PROMPT, "有「转人工」的兜底说明")
    check(
        "不要" in SYSTEM_PROMPT and ("编" in SYSTEM_PROMPT or "凭常识" in SYSTEM_PROMPT),
        "有「不许编」的约束",
        "这条是 v2 那个 A/B 实验的直接产物：一放宽，编造从 3 条涨到 14 条",
    )
    check(
        "路径" in SYSTEM_PROMPT and "让你去做" in SYSTEM_PROMPT,
        "有「问怎么做 ≠ 让你去做」的区分",
        "第 8 条。#25 就是栽在这上面：问入口路径，它答「我帮你转」",
    )


# ============================================================
# 4.5 护栏：规则类问题不许裸答
# ============================================================


def check_guard() -> None:
    from src.agent import cs_agent

    section(5, 7, "护栏 4：规则类问题不许裸答")

    check(
        hasattr(cs_agent, "_NUDGE_TEXT") and hasattr(cs_agent, "_RULE_Q_RE"),
        "护栏 4 存在：规则类问题没查资料就想答，会被打回一次",
        "prompt 是软约束，模型会不听。这一条是代码兜底",
    )

    src = inspect.getsource(cs_agent.CSAgent.run)
    check(
        "nudged" in src and "retrieved" in src,
        "打回只执行一次，且只认「成功查到」",
        "retrieved 不认失败的调用 —— 否则知识库没加载时这条护栏就废了",
    )


# ============================================================
# 5. 判例库的防泄漏
# ============================================================


def check_leak_guard() -> None:
    from src.agent.retrieval import PrecedentIndex

    section(6, 7, "防答案泄漏")

    sig = inspect.signature(PrecedentIndex.search)
    p = sig.parameters.get("exclude_id")
    if p is None:
        check(False, "search() 有 exclude_id 参数")
        return

    required = p.default is inspect.Parameter.empty
    check(
        required,
        "exclude_id 是必传参数（没有默认值）",
        "有默认值就能被悄悄省略，评测数字会好看得离谱且毫无意义",
    )


# ============================================================
# 6. 知识和判例的分工有没有混
# ============================================================


def check_separation() -> None:
    from src.agent.knowledge import KnowledgeBase
    from src.agent.retrieval import PrecedentIndex

    section(7, 7, "知识 / 判例分工")

    kb_sig = inspect.signature(KnowledgeBase.search)
    check(
        "exclude_id" not in kb_sig.parameters,
        "知识库不参与 leave-one-out（没有 exclude_id）",
        "它是规则来源，不是测试集的一部分，不需要排除自己",
    )

    kb = KnowledgeBase()
    has_answer_field = any("answer" in r for r in kb.rules)
    check(
        not has_answer_field,
        "知识库条目里没有 answer 字段",
        "有的话说明有人把判例塞进知识库了，两者的角色会混掉",
    )

    pc_sig = inspect.signature(PrecedentIndex.search)
    check("exclude_id" in pc_sig.parameters, "判例库参与 leave-one-out")


# ============================================================
# 入口
# ============================================================


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    table = "--table" in argv

    print("AI 客服评测台 · 自检")
    print("不联网、不花钱、几秒跑完。全过了再去跑真评测。")

    checks = [
        check_knowledge_file,
        lambda: check_retrieval(table=table),
        check_tools,
        check_prompt,
        check_guard,
        check_leak_guard,
        check_separation,
    ]

    for fn in checks:
        try:
            fn()
        except Exception as exc:                   # noqa: BLE001
            check(False, f"{fn.__name__} 执行出错", f"{type(exc).__name__}: {exc}")

    passed = sum(1 for ok, _, _ in results if ok)
    total = len(results)
    failed = [(t, d) for ok, t, d in results if not ok]

    print("\n" + "=" * 56)
    print(f"结论：{passed}/{total} 项通过")
    if failed:
        print("\n没过的项：")
        for t, d in failed:
            print(f"  - {t}" + (f"（{d}）" if d else ""))
        print("\n先去修这些，再花 API 额度跑评测。")
        return 1

    print("代码状态符合预期。下一步：")
    print("  .venv/Scripts/python.exe -m src.eval.runner    # 真跑，约 1-2 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
