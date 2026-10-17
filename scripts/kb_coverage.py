"""
知识库覆盖检查：新一批问题，知识库里到底有没有对应的规则。

为什么需要这个东西：

    项目现在 30/30，但那是**开卷** —— 30 条规则就是从这 30 条实测记录里提炼的，
    一问一答对得上号。换一批新问题进来，第一个要回答的不是「Agent 答得怎么样」，
    而是「知识库里有没有这条」。

    没有这条的时候，Agent 不会告诉你「我不知道」，它会：
      · 拿一条不相关的规则照着答（最危险，答得还挺像）
      · 或者转人工（安全，但问题没解决）

    所以导入新报告的第一步，不是跑评测，是跑这个。

用法（项目根目录）：

    # 用现有的 30 条评测问题
    .venv/Scripts/python.exe scripts/kb_coverage.py

    # 自己的一批新问题，一行一条
    .venv/Scripts/python.exe scripts/kb_coverage.py --file 新问题.txt

    # 每题多看几条候选，判断「有没有一条勉强能对上」
    .venv/Scripts/python.exe scripts/kb_coverage.py --file 新问题.txt --top 3

一个必须说在前面的局限：**分数不能单独当判据。**
留出实验里，「转人工客服要怎么操作」命中「实名信息填错」得 10.46，Agent 靠常识答对了；
「自动续费扣的钱能退吗」命中「关闭自动续费」得 30.23，看着更像，却答错了。
BM25 算的是字面相似，算不出「是不是同一件事」。所以这张表要人看主题，不要只看分数。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 两个经验值，都来自留出实验：
#   · 30 条问题命中**正确**主题时，得分最低是 25.73。
#   · 命中错误主题时，得分可以低到 9.22，也可以高到 30.23。
# 所以这不是判据，只是「什么时候该多看几个候选」的提示线。
ATTN = 25.0


def load_questions(argv: list[str]) -> list[str]:
    if "--file" in argv:
        i = argv.index("--file")
        if i + 1 >= len(argv):
            raise SystemExit("--file 后面要跟文件路径")
        p = Path(argv[i + 1])
        if not p.exists():
            raise SystemExit(f"找不到 {p}")
        lines = p.read_text(encoding="utf-8").splitlines()
        qs = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
        if not qs:
            raise SystemExit("文件里没有有效的问题（空行和 # 开头的会被跳过）")
        return qs

    from src.config import RECORDS_JSON

    import json

    recs = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    recs = recs if isinstance(recs, list) else recs["records"]
    return [r["question"] for r in recs]


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    top = 3 if ("--top" in argv and "3" in argv) else 1

    from src.agent.knowledge import KnowledgeBase

    try:
        kb = KnowledgeBase()
    except FileNotFoundError as e:
        print(f"[X] 知识库没找到：{e}")
        return 1

    qs = load_questions(argv)

    print("=" * 78)
    print(f"知识库覆盖检查：{len(qs)} 条问题　知识库 {len(kb.rules)} 条规则")
    print("=" * 78)
    print()

    rows = []
    for q in qs:
        # 至少取 3 条：低分时要展开备选，只查 1 条就展不开了。
        hits = kb.search(q, k=max(top, 3))
        rows.append((hits, q))

    # 分数低的排前面 —— 那是最可能没覆盖的
    rows.sort(key=lambda r: r[0][0]["score"] if r[0] else 0)

    suspect = 0
    for hits, q in rows:
        if not hits:
            print(f"  {'—':>7}  {q[:30]:<32} 　（零命中）")
            suspect += 1
            continue
        head = hits[0]
        weak = head["score"] < ATTN
        mark = "⚠️" if weak else "  "
        if weak:
            suspect += 1
        print(f"{mark}{head['score']:>7.2f}  {q[:30]:<32} → {head['topic']}")
        # 分数不够高的时候把备选也打出来 —— 真正对的规则经常在第二、三名，
        # 只看到 top1 会误以为「没覆盖」。
        if weak or top > 1:
            for h in hits[1:]:
                print(f"  {'':>7}  {'':<32}   （备选 {h['score']:.2f}）{h['topic']}")

    scores = [h[0]["score"] for h, _ in rows if h]
    print()
    print("-" * 78)
    if scores:
        print(
            f"得分区间 {min(scores):.2f} ~ {max(scores):.2f}　"
            f"中位数 {sorted(scores)[len(scores) // 2]:.2f}"
        )
    print(f"⚠️ 标记 {suspect} 条（得分 < {ATTN}，已展开备选）")

    print(
        "\n怎么读这张表：\n"
        "  不看分数，看箭头右边的主题和左边的问题是不是同一件事。\n"
        "  「是同一件事」→ 知识库覆盖了，Agent 有依据可答。\n"
        "  「不是同一件事」→ 这条要补规则。补法：按 topic 的粒度写一条，\n"
        "    包含结论、分档条件、时限、操作路径四样，写进 data/knowledge/cs_rules.json。\n"
        "\n  分数低只说明字面不像，不代表一定没覆盖；分数高也可能撞上无关规则。\n"
        "  留出实验里 10.46 分的那条答对了，30.23 分的那条答错了 —— 所以别指望阈值。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
