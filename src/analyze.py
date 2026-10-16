"""
[M3 + M4] 融合评分与一致性验证

职责：
    1. 把规则分和大模型分按权重合成「最终分」
    2. 把机器的最终判定和人工标注逐条比对
    3. 算混淆矩阵和 Cohen's Kappa，量化「这套工具到底有多可信」

为什么这一步是整个项目的技术亮点：
    规则分和 LLM 分各自都有毛病 —— 规则抓不到语义问题（第 23 条漏检就是证据），
    大模型又会飘。合起来用还不够，关键是要**证明**合起来到底靠不靠谱。
    Kappa 就是那个证明。

运行方式（项目根目录）：
    python -m src.analyze

------------------------------------------------------------------
【DAY 4 任务】下面四个函数留给你实现，都标了 TODO。
要用的公式、每一步该算什么，都写在 docstring 里。
卡住了随时问，不要硬憋。
------------------------------------------------------------------
"""

import json
import sys

from src.config import (
    OUTPUT_DIR,
    PASS_THRESHOLD,
    RECORDS_JSON,
    WEIGHT_LLM,
    WEIGHT_RULE,
)

RULE_RESULTS = OUTPUT_DIR / "rule_results.json"
LLM_RESULTS = OUTPUT_DIR / "llm_results.json"


# ============================================================
# 一、加载（已写好，不用动）
# ============================================================


def load_all() -> tuple[list[dict], dict, dict]:
    """把三份数据读进来：原始记录、规则评分、大模型评分。

    返回 (records, rule_map, llm_map)，后两个是 {id: 结果} 的字典，方便按 id 查。
    """
    records = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    rule_list = json.loads(RULE_RESULTS.read_text(encoding="utf-8"))
    llm_list = json.loads(LLM_RESULTS.read_text(encoding="utf-8"))

    rule_map = {r["id"]: r for r in rule_list}
    llm_map = {r["id"]: r for r in llm_list}
    return records, rule_map, llm_map


# ============================================================
# 二、★ TODO 1：混合评分
# ============================================================


def blend_score(rule_score: float, llm_score: float) -> float:
    """把规则分和大模型分加权合成最终分。

    公式：
        final = WEIGHT_RULE × rule_score + WEIGHT_LLM × llm_score

    两个权重都在 config.py 里（0.4 和 0.6），**别写死数字，从 config 导入**。
    返回值保留 2 位小数。

    举个例子验证你写对了：
        规则分 2.4、大模型分 2 → 0.4×2.4 + 0.6×2 = 0.96 + 1.2 = 2.16 → 2.16
    """
    rule_part = WEIGHT_RULE * rule_score      # 规则分的贡献
    llm_part = WEIGHT_LLM * llm_score         # 大模型分的贡献
    return round(rule_part + llm_part, 2)     # 加权求和，保留两位小数


def blend_all(records: list[dict], rule_map: dict, llm_map: dict) -> list[dict]:
    """对全部记录做合成，返回 [{id, rule_score, llm_score, final_score, final_label, ...}]。

    提示：
        - rule_score 从 rule_map[rec["id"]]["rule_score"] 取
        - llm_score 从 llm_map[rec["id"]]["score"] 取
        - 模型评审失败的条目（llm_map 里没有）要跳过，别让程序崩
        - 判定规则：final_score >= PASS_THRESHOLD 就是「答对」
    """
    results = []
    for rec in records:
        rid = rec["id"]

        # 大模型评审失败、或规则没跑到的条目，直接跳过，不让它污染统计
        if rid not in llm_map or rid not in rule_map:
            continue

        rule_score = rule_map[rid]["rule_score"]
        llm_score = llm_map[rid]["score"]
        final_score = blend_score(rule_score, llm_score)

        results.append(
            {
                "id": rid,
                "rule_score": rule_score,
                "llm_score": llm_score,
                "final_score": final_score,
                # 条件表达式：达到阈值算答对，否则答偏
                "final_label": "答对" if final_score >= PASS_THRESHOLD else "答偏",
                # 人工标注从原始记录里取，用来做后面的比对
                "human_label": rec["human_label"],
            }
        )

    return results


# ============================================================
# 三、★ TODO 2：一致率与混淆矩阵
# ============================================================


def build_pairs(blended: list[dict], records: list[dict]) -> list[tuple[int, str, str]]:
    """把每条记录配成 (id, 机器判定, 人工判定) 的三元组。

    为什么必须带上 id：
        后面算「漏检」时要报告「是哪几条被判错了」，光有判定结果没法定位到具体条目。

    返回示例：
        [(1, "答偏", "答偏"), (2, "答对", "答对"), (3, "答对", "答偏"), ...]

    提示：
        - 机器判定在 blended 里，字段名 final_label
        - 人工判定在 records 里，字段名 human_label
        - 两边按 id 对齐
    """
    # 先做一个「id → 人工判定」的字典，这样按 id 查人工标注是 O(1)，不用每次遍历列表
    human_by_id = {r["id"]: r["human_label"] for r in records}

    pairs = []
    for b in blended:
        pairs.append((b["id"], b["final_label"], human_by_id[b["id"]]))

    return pairs


def agreement(pairs: list[tuple[int, str, str]]) -> dict:
    """算一致率和混淆矩阵。

    返回这个字典：
        {
            "total": 总条数,
            "correct": 一致的条数,
            "accuracy": 一致率（保留 4 位小数）,
            "miss": [漏检的 id 列表],          # 人判「答偏」、机器判「答对」
            "false_alarm": [误伤的 id 列表],   # 人判「答对」、机器判「答偏」
            "matrix": {
                "机器答对_人工答对": 数量,
                "机器答对_人工答偏": 数量,
                "机器答偏_人工答对": 数量,
                "机器答偏_人工答偏": 数量,
            },
        }

    「漏检」和「误伤」这两个概念在 D2 的 rules.py 里已经出现过一次，
    可以回去看 compare_with_human 是怎么写的，思路一样。
    """
    total = len(pairs)
    correct = 0
    miss: list[int] = []          # 漏检：机器放过，人工判偏
    false_alarm: list[int] = []   # 误伤：机器判偏，人工判对

    matrix = {
        "机器答对_人工答对": 0,
        "机器答对_人工答偏": 0,
        "机器答偏_人工答对": 0,
        "机器答偏_人工答偏": 0,
    }

    for pid, machine, human in pairs:
        if machine == human:
            correct += 1

        key = f"机器{machine}_人工{human}"
        if key in matrix:
            matrix[key] += 1

        if machine == "答对" and human == "答偏":
            miss.append(pid)
        elif machine == "答偏" and human == "答对":
            false_alarm.append(pid)

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "miss": miss,
        "false_alarm": false_alarm,
        "matrix": matrix,
    }


# ============================================================
# 四、★ TODO 3：Cohen's Kappa（本项目的核心指标）
# ============================================================


def cohen_kappa(pairs: list[tuple[int, str, str]]) -> float:
    """计算 Cohen's Kappa 系数。保留 4 位小数。

    注意：传进来的 pairs 是三元组 (id, 机器判定, 人工判定)。
    算 Kappa 只需要后两个元素 —— 即 pair[1] 和 pair[2]。

    ------------------------------------------------------------
    为什么要算这个，而不是只看准确率？

        假设 30 条里有 21 条答对、9 条答偏。
        那么一个「闭着眼睛全判答对」的傻瓜分类器，准确率也有 21/30 = 70%。
        准确率高的假象，可能只是因为它押中了多数类。
        Kappa 的作用就是把「瞎猜也能蒙对」的部分扣掉，
        剩下的才是工具真正的判断力。

    ------------------------------------------------------------
    公式：

        κ = (Po − Pe) / (1 − Pe)

        Po = 观测一致率 = 一致的条数 / 总条数
             （也就是 agreement() 里算出来的 accuracy）

        Pe = 期望一致率（纯随机情况下，两边碰巧一致的期望概率）
           对于只有「答对 / 答偏」两个类别的情况：
           Pe = 机器判答对的比例 × 人工判答对的比例
              + 机器判答偏的比例 × 人工判答偏的比例

    ------------------------------------------------------------
    手算示例（3 条，2 条一致）：
        pairs = [(1,"答对","答对"), (2,"答对","答对"), (3,"答对","答偏")]

        Po = 2/3 ≈ 0.6667
        机器判答对比例 = 3/3 = 1.0，人工判答对比例 = 2/3 ≈ 0.6667
        机器判答偏比例 = 0/3 = 0.0，人工判答偏比例 = 1/3 ≈ 0.3333
        Pe = 1.0 × 0.6667 + 0.0 × 0.3333 = 0.6667

        κ = (0.6667 − 0.6667) / (1 − 0.6667) = 0

        解读：虽然准确率有 66.7%，但 Kappa = 0 —— 因为我们那个「全判答对」
        的策略并没有比瞎猜更好。这就是 Kappa 的价值。

    ------------------------------------------------------------
    解读标准（这是通用的业界口径，可以写进报告）：
        κ > 0.80      几乎完全一致
        0.60 – 0.80   高度一致
        0.40 – 0.60   中等一致
        < 0.40        一致性偏弱，工具的判定需要重新校准
    """
    total = len(pairs)
    if total == 0:
        return 0.0

    # ---------- 第一步：观测一致率 Po ----------
    # pairs 是三元组，p[1] 是机器判定，p[2] 是人工判定
    agree_count = sum(1 for p in pairs if p[1] == p[2])
    po = agree_count / total

    # ---------- 第二步：期望一致率 Pe ----------
    # 机器和人工各自判「答对」的比例（答偏比例用 1 减出来）
    machine_ok = sum(1 for p in pairs if p[1] == "答对") / total
    human_ok = sum(1 for p in pairs if p[2] == "答对") / total
    machine_bad = 1 - machine_ok
    human_bad = 1 - human_ok

    # 纯随机情况下碰巧一致的期望概率：两种「都答对」「都答偏」的情况相加
    pe = machine_ok * human_ok + machine_bad * human_bad

    # 极端情况：两边都只会判同一个标签，此时 Pe = 1，分母为 0，无意义
    if pe >= 1:
        return 1.0

    # ---------- 第三步：套公式 ----------
    kappa = (po - pe) / (1 - pe)
    return round(kappa, 4)


# ============================================================
# 五、报告输出（已写好，不用动）
# ============================================================


def print_report(blended: list[dict], agg: dict, kappa: float) -> None:
    print("\n" + "-" * 58)
    print("混合评分结果")
    print(f"  权重：规则 {WEIGHT_RULE} / 大模型 {WEIGHT_LLM}")
    print(f"  判定阈值：{PASS_THRESHOLD}")
    print("-" * 58)

    print(f"\n  逐条明细（{len(blended)} 条）\n")
    print(f"  {'序号':<6}{'规则分':<9}{'大模型分':<11}{'最终分':<10}{'机器判定':<11}{'人工'}")

    for b in blended:
        mark = "" if b["final_label"] == b["human_label"] else "   ← 不一致"
        print(f"  {b['id']:<7}{b['rule_score']:<10}{b['llm_score']:<12}"
              f"{b['final_score']:<11}{b['final_label']:<12}{b['human_label']}{mark}")

    print("\n" + "-" * 58)
    print("一致性验证")
    print(f"  总条数            {agg['total']}")
    print(f"  判定一致          {agg['correct']}")
    print(f"  一致率            {agg['accuracy']:.1%}")
    print(f"  漏检（人偏机对）  {agg['miss'] or '无'}")
    print(f"  误伤（人对机偏）  {agg['false_alarm'] or '无'}")

    print("\n  混淆矩阵")
    for key, val in agg["matrix"].items():
        print(f"    {key:<20}{val}")

    print(f"\n  Cohen's Kappa     {kappa:.4f}")

    if kappa > 0.8:
        level = "几乎完全一致 —— 工具判定高度可信"
    elif kappa > 0.6:
        level = "高度一致 —— 工具判定可用"
    elif kappa > 0.4:
        level = "中等一致 —— 可用于初筛，但需人工复核"
    else:
        level = "一致性偏弱 —— 判定标准需要重新校准"
    print(f"  解读              {level}")
    print("-" * 58)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 58)
    print("[M3+M4] 融合评分与一致性验证")
    print("=" * 58)

    print("\n[1/4] 加载数据 ...")
    try:
        records, rule_map, llm_map = load_all()
    except FileNotFoundError as e:
        print(f"      [X] 缺文件：{e}")
        print("      请先跑完 src.rules 和 src.llm_judge，确保 output/ 下两份结果都在。")
        return 1
    print(f"      记录 {len(records)} 条 · 规则分 {len(rule_map)} 条 · 大模型分 {len(llm_map)} 条")

    if len(llm_map) < len(records):
        print(f"      [!] 大模型评审只完成了 {len(llm_map)}/{len(records)} 条，"
              f"本次只比对已完成的 {len(llm_map)} 条")

    print("\n[2/4] 合成混合评分 ...")
    try:
        blended = blend_all(records, rule_map, llm_map)
    except NotImplementedError as e:
        print(f"      [  ] 还没实现：{e}")
        return 2
    print(f"      完成 {len(blended)} 条")

    print("\n[3/4] 比对人工标注 ...")
    pairs = build_pairs(blended, records)
    agg = agreement(pairs)

    print("\n[4/4] 计算 Cohen's Kappa ...")
    kappa = cohen_kappa(pairs)

    print_report(blended, agg, kappa)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "final_results.json"
    out.write_text(
        json.dumps(
            {"blended": blended, "agreement": agg, "kappa": kappa},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已写出：{out.name}")
    print("\n下一步：DAY 5 用 Streamlit 把这一屏做成可视化界面")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
