"""
[M5] Streamlit 可视化界面 —— AI 客服质量评测台

把前面四个模块的分析结果做成一块可视化看板：
    · 顶部四张指标卡：评测轮数 / 机器答偏率 / 人机一致率 / Cohen's Kappa
    · 中间两张分布图：错因分布、评分分布
    · 底部逐条明细表，人机判定不一致的行会标出来
    · 侧栏可实时调「规则分 vs 大模型分」的权重和判定阈值

为什么侧栏能实时调参：
    规则分和大模型分都已经落盘（rule_results.json / llm_results.json），
    改权重只是重新做一次加权求和，**不需要再调用一次大模型 API**。
    演示时拖动滑块、看着一致率和 Kappa 变化，是这套工具最有说服力的地方。

运行方式（项目根目录，不是 python 而是 streamlit）：
    streamlit run app.py

停止：在终端按 Ctrl + C
"""

import json

import pandas as pd
import streamlit as st

from src.analyze import agreement, cohen_kappa
from src.config import (
    MAX_SCORE,
    OUTPUT_DIR,
    PASS_THRESHOLD,
    RECORDS_JSON,
    WEIGHT_RULE,
)

# 这一行必须是第一个 streamlit 调用
st.set_page_config(page_title="AI 客服质量评测台", layout="wide")


# ============================================================
# 数据加载
# ============================================================


@st.cache_data
def load_json(path_str: str):
    """读 JSON。用 cache_data 包一层，避免每次交互都重新读盘。"""
    return json.loads(open(path_str, encoding="utf-8").read())


def load_inputs():
    """把三份输入读进来：原始记录、规则分、大模型分。"""
    rule_path = OUTPUT_DIR / "rule_results.json"
    llm_path = OUTPUT_DIR / "llm_results.json"

    missing = [p.name for p in (RECORDS_JSON, rule_path, llm_path) if not p.exists()]
    if missing:
        st.error(
            f"缺少结果文件：{', '.join(missing)}\n\n"
            "请先在终端依次运行：\n"
            "```\npython -m src.loader\npython -m src.rules\npython -m src.llm_judge\n```"
        )
        st.stop()

    records = load_json(str(RECORDS_JSON))
    rule_map = {r["id"]: r for r in load_json(str(rule_path))}
    llm_map = {r["id"]: r for r in load_json(str(llm_path))}
    return records, rule_map, llm_map


# ============================================================
# 核心计算：给定权重和阈值，重算全部指标
# ============================================================


def recompute(records, rule_map, llm_map, w_rule, w_llm, threshold):
    """按当前权重和阈值，把 30 条重新合成一遍并算指标。"""
    rows = []
    for rec in records:
        rid = rec["id"]
        if rid not in rule_map or rid not in llm_map:
            continue

        rs = rule_map[rid]["rule_score"]
        ls = llm_map[rid]["score"]
        final = round(w_rule * rs + w_llm * ls, 2)

        rows.append(
            {
                "序号": rid,
                "场景": rec["scene"],
                "提问": rec["question"],
                "规则分": rs,
                "大模型分": ls,
                "最终分": final,
                "机器判定": "答对" if final >= threshold else "答偏",
                "人工标注": rec["human_label"],
                "大模型理由": llm_map[rid]["reason"],
                "大模型标签": llm_map[rid]["reason_tag"],
            }
        )

    df = pd.DataFrame(rows)

    pairs = [
        (r["序号"], r["机器判定"], r["人工标注"])
        for _, r in df.iterrows()
    ]
    agg = agreement(pairs)
    kappa = cohen_kappa(pairs)

    return df, agg, kappa


# ============================================================
# 页面
# ============================================================


st.title("AI 客服质量评测台")
st.caption(
    "规则引擎 + 大模型评审混合评分 · 以 30 轮人工标注为基准验证机器判定的可信度"
)

records, rule_map, llm_map = load_inputs()

# ---------- 侧栏：可调参数 ----------
with st.sidebar:
    st.header("评分参数")

    w_rule = st.slider(
        "规则分权重",
        min_value=0.0,
        max_value=1.0,
        value=float(WEIGHT_RULE),
        step=0.05,
        help="规则分偏「表面特征」，大模型分偏「语义理解」。两者权重之和为 1。",
    )
    w_llm = round(1.0 - w_rule, 2)
    st.caption(f"大模型分权重自动为 {w_llm}")

    threshold = st.slider(
        "判定阈值",
        min_value=1.0,
        max_value=5.0,
        value=float(PASS_THRESHOLD),
        step=0.1,
        help="最终分达到这个值就判为「答对」，否则「答偏」。",
    )

    st.divider()
    st.caption(f"评分满分：{MAX_SCORE} 分")
    st.caption("数据来源：30 轮某第三方支付平台的在线客服实测记录")
    st.caption("人工标注：答对 21 / 答偏 9")

df, agg, kappa = recompute(records, rule_map, llm_map, w_rule, w_llm, threshold)

# ---------- 指标卡 ----------
machine_bad = int((df["机器判定"] == "答偏").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("评测轮数", f"{agg['total']}")
c2.metric(
    "机器判答偏",
    f"{machine_bad} 条",
    delta=f"{machine_bad / agg['total']:.1%}",
    delta_color="inverse",   # 答偏率高是坏事，配色要反过来
)
c3.metric("人机一致率", f"{agg['accuracy']:.1%}", delta=f"{agg['correct']}/{agg['total']} 条一致")
c4.metric("Cohen's Kappa", f"{kappa:.4f}")

if kappa > 0.8:
    st.success(f"Kappa = {kappa:.4f}，属「几乎完全一致」——机器判定高度可信。")
elif kappa > 0.6:
    st.info(f"Kappa = {kappa:.4f}，属「高度一致」——机器判定可用。")
elif kappa > 0.4:
    st.warning(f"Kappa = {kappa:.4f}，属「中等一致」——只能做初筛，需人工复核。")
else:
    st.error(f"Kappa = {kappa:.4f}，一致性偏弱——判定标准需要重新校准。")

# ---------- 分布图 ----------
st.subheader("分布情况")

g1, g2 = st.columns(2)

with g1:
    st.caption("人工标注 vs 机器判定（按场景）")
    scene_cross = (
        df.groupby(["场景", "机器判定"]).size().unstack(fill_value=0)
    )
    st.bar_chart(scene_cross)

with g2:
    st.caption("大模型评分分布（0–5 分）")
    score_dist = (
        df["大模型分"].value_counts().sort_index().rename_axis("评分").to_frame("条数")
    )
    st.bar_chart(score_dist)

# ---------- 错因分布 ----------
st.subheader("错因标签对比")

t1, t2 = st.columns(2)

# 只统计被判为「答偏」的条目。
# 如果把 21 条「不适用」也画进来，两类各自的柱子都会被压成一根高柱，看不出真实差异。
human_bad = [r["human_reason"] for r in records if r["human_label"] == "答偏"]

with t1:
    st.caption(f"人工标注的错因（{len(human_bad)} 条答偏）")
    human_reason = (
        pd.Series(human_bad).value_counts().rename_axis("错因").to_frame("条数")
    )
    st.bar_chart(human_reason)

with t2:
    machine_bad_tags = df[df["机器判定"] == "答偏"]["大模型标签"].tolist()
    st.caption(f"大模型给出的错因（{len(machine_bad_tags)} 条判偏）")
    llm_reason = (
        pd.Series(machine_bad_tags).value_counts().rename_axis("错因").to_frame("条数")
    )
    st.bar_chart(llm_reason)

st.caption(
    "注：两者的差异本身就是一个有价值的发现——大模型判断「答对/答偏」很准，"
    "但判断「错在哪一类」时不够稳定。"
)

# ---------- 不一致条目 ----------
st.subheader("人机判定不一致的条目")

diff = df[df["机器判定"] != df["人工标注"]]
if diff.empty:
    st.success("当前参数下，机器判定与人工标注完全一致。")
else:
    for _, r in diff.iterrows():
        kind = "漏检（机器放过，人工判偏）" if r["机器判定"] == "答对" else "误伤（机器判偏，人工判对）"
        with st.expander(f"第 {r['序号']} 条 · {kind} · 最终分 {r['最终分']}"):
            st.write(f"**提问**：{r['提问']}")
            st.write(f"**评分**：规则 {r['规则分']} 分 / 大模型 {r['大模型分']} 分 → 最终 {r['最终分']} 分")
            st.write(f"**大模型理由**：{r['大模型理由']}")
            st.write(f"**大模型标签**：{r['大模型标签']}")

# ---------- 明细表 ----------
st.subheader(f"逐条明细（{len(df)} 条）")

show = df[
    ["序号", "场景", "提问", "规则分", "大模型分", "最终分", "机器判定", "人工标注", "大模型标签"]
].copy()
show["是否一致"] = show.apply(
    lambda r: "一致" if r["机器判定"] == r["人工标注"] else "不一致", axis=1
)


def mark_row(row):
    """不一致的行标浅红，机器判偏的标浅黄。"""
    if row["是否一致"] == "不一致":
        return ["background-color: #FCEBEB"] * len(row)
    if row["机器判定"] == "答偏":
        return ["background-color: #FAEEDA"] * len(row)
    return [""] * len(row)


st.dataframe(
    show.style.apply(mark_row, axis=1),
    width="stretch",
    hide_index=True,
    height=520,
)

st.divider()
st.caption(
    "混合评分公式：最终分 = 规则分权重 × 规则分 + 大模型分权重 × 大模型分。"
    "拖动侧栏滑块可实时重算，无需重新调用大模型接口。"
)
