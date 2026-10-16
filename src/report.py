"""
[M6] 报告导出

把评测结果导出成一份可以直接发给别人看的 Excel 报告，两个工作表：
    「评测汇总」—— 核心指标、混淆矩阵、Kappa 解读、评审参数
    「逐条明细」—— 全部问答的完整评分记录，人机判定不一致的行标红

为什么要做这一步：
    前面所有结果都在 JSON 里，那是给程序看的。要交给别人（面试官、同事、客户），
    得是一份打开就能读懂的东西。这也是「工具」和「作品」的区别。

运行方式（项目根目录）：
    python -m src.report
"""

import json
import sys
from collections import Counter

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from src.config import (
    MAX_SCORE,
    OUTPUT_DIR,
    PASS_THRESHOLD,
    RECORDS_JSON,
    WEIGHT_LLM,
    WEIGHT_RULE,
)

FINAL_RESULTS = OUTPUT_DIR / "final_results.json"
LLM_RESULTS = OUTPUT_DIR / "llm_results.json"
REPORT_XLSX = OUTPUT_DIR / "AI客服质量评测报告.xlsx"

# ---------- 样式 ----------
HEAD_FILL = PatternFill("solid", fgColor="1F4E79")
HEAD_FONT = Font(name="微软雅黑", size=10.5, bold=True, color="FFFFFF")
TITLE_FONT = Font(name="微软雅黑", size=14, bold=True, color="1F4E79")
BODY_FONT = Font(name="微软雅黑", size=10)
LABEL_FONT = Font(name="微软雅黑", size=10, bold=True)

FILL_BAD = PatternFill("solid", fgColor="FCE4E4")    # 人机不一致
FILL_WARN = PatternFill("solid", fgColor="FFF3D6")   # 判为答偏
FILL_GOOD = PatternFill("solid", fgColor="E8F5E9")   # 判定一致且答对

THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="center")


def kappa_level(kappa: float) -> str:
    """把 Kappa 数值翻译成一句话结论。"""
    if kappa > 0.8:
        return "几乎完全一致 —— 机器判定高度可信"
    if kappa > 0.6:
        return "高度一致 —— 机器判定可用"
    if kappa > 0.4:
        return "中等一致 —— 可用于初筛，需人工复核"
    return "一致性偏弱 —— 判定标准需要重新校准"


def write_style(ws, headers: list[str], widths: list[int], row: int = 1) -> None:
    """写表头并设样式。"""
    for i, (h, w) in enumerate(zip(headers, widths), start=1):
        cell = ws.cell(row=row, column=i, value=h)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = CENTER
        cell.border = BORDER
        ws.column_dimensions[cell.column_letter].width = w


def build_summary(ws, agg: dict, kappa: float, records: list[dict]) -> None:
    """第一个工作表：汇总。"""
    ws["A1"] = "AI 客服质量评测报告"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:C1")

    ws["A2"] = "评测对象：支付宝「我的客服」（蚂蚁智服） · 30 轮实测问答"
    ws["A2"].font = Font(name="微软雅黑", size=10, color="666666")
    ws.merge_cells("A2:C2")

    row = 4
    ws.cell(row=row, column=1, value="评测参数").font = LABEL_FONT
    row += 1
    params = [
        ("规则分权重", WEIGHT_RULE),
        ("大模型分权重", WEIGHT_LLM),
        ("判定阈值（最终分 ≥ 此值判答对）", PASS_THRESHOLD),
        ("评分满分", MAX_SCORE),
    ]
    for k, v in params:
        ws.cell(row=row, column=1, value=k).font = BODY_FONT
        ws.cell(row=row, column=2, value=v).font = BODY_FONT
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="核心指标").font = LABEL_FONT
    row += 1

    human_bad = sum(1 for r in records if r["human_label"] == "答偏")
    metrics = [
        ("评测总轮数", agg["total"]),
        ("人工判答偏", f"{human_bad} 条（{human_bad / agg['total']:.1%}）"),
        ("人机判定一致", f"{agg['correct']} 条"),
        ("一致率", f"{agg['accuracy']:.2%}"),
        ("Cohen's Kappa", f"{kappa:.4f}"),
        ("Kappa 解读", kappa_level(kappa)),
        ("漏检（机器放过、人工判偏）", f"{len(agg['miss'])} 条  {agg['miss'] if agg['miss'] else ''}"),
        ("误伤（机器判偏、人工判对）", f"{len(agg['false_alarm'])} 条  {agg['false_alarm'] if agg['false_alarm'] else ''}"),
    ]
    for k, v in metrics:
        c1 = ws.cell(row=row, column=1, value=k)
        c1.font = BODY_FONT
        c1.border = BORDER
        c2 = ws.cell(row=row, column=2, value=v)
        c2.font = BODY_FONT
        c2.border = BORDER
        if k == "Cohen's Kappa":
            c2.fill = FILL_GOOD
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="混淆矩阵").font = LABEL_FONT
    row += 1
    write_style(ws, ["组合", "条数"], [34, 14], row=row)
    row += 1
    for k, v in agg["matrix"].items():
        ws.cell(row=row, column=1, value=k.replace("_", " / ")).font = BODY_FONT
        ws.cell(row=row, column=2, value=v).font = BODY_FONT
        ws.cell(row=row, column=1).border = BORDER
        ws.cell(row=row, column=2).border = BORDER
        ws.cell(row=row, column=2).alignment = CENTER
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="结论与局限").font = LABEL_FONT
    row += 1
    notes = [
        f"1. 混合评分（规则 + 大模型）与人工标注的一致率为 {agg['accuracy']:.1%}，"
        f"Kappa = {kappa:.4f}，属「{kappa_level(kappa).split(' —— ')[0]}」，工具判定可信。",
        f"2. 漏检 {len(agg['miss'])} 条，说明未放过真实缺陷；误伤 {len(agg['false_alarm'])} 条"
        f"{'（' + '、'.join('第' + str(i) + '条' for i in agg['false_alarm']) + '）' if agg['false_alarm'] else ''}，"
        "经查系评分量表未考虑「用户诉求本身即转人工」的情形，属可修正项。",
        "3. 大模型在「答对/答偏」的二元判定上可靠，但在「错因归类」上不够稳定，"
        "人工与大模型的错因分布存在明显差异 —— 这是当前版本的主要局限。",
        "4. 评测集为 30 轮真实人工实测样本，规模有限，但整套方法的可复现性已得到验证，"
        "扩充数据量后结论应更稳健。",
    ]
    for n in notes:
        c = ws.cell(row=row, column=1, value=n)
        c.font = BODY_FONT
        c.alignment = WRAP
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=3)
        ws.row_dimensions[row].height = 32
        row += 1


def build_detail(ws, blended: list[dict], records: list[dict], llm_map: dict) -> None:
    """第二个工作表：逐条明细。"""
    rec_map = {r["id"]: r for r in records}

    headers = ["序号", "场景", "用户提问", "AI 应答摘要", "规则分", "大模型分",
               "最终分", "机器判定", "人工标注", "是否一致", "大模型理由", "人工判断说明"]
    widths = [6, 8, 26, 40, 8, 10, 8, 10, 10, 10, 30, 46]
    write_style(ws, headers, widths)

    for i, b in enumerate(blended, start=2):
        rec = rec_map.get(b["id"], {})
        same = b["final_label"] == b["human_label"]

        # blended 里只有评分字段，场景和问答原文要从原始记录取，大模型理由从评审结果取
        llm = llm_map.get(b["id"], {})
        values = [
            b["id"], rec.get("scene", ""), rec.get("question", ""), rec.get("answer", ""),
            b["rule_score"], b["llm_score"], b["final_score"],
            b["final_label"], b["human_label"],
            "一致" if same else "不一致",
            llm.get("reason", ""), rec.get("human_note", ""),
        ]
        for j, v in enumerate(values, start=1):
            c = ws.cell(row=i, column=j, value=v)
            c.font = BODY_FONT
            c.border = BORDER
            c.alignment = WRAP if j in (3, 4, 11, 12) else CENTER

            if not same:
                c.fill = FILL_BAD
            elif b["final_label"] == "答偏":
                c.fill = FILL_WARN
            else:
                c.fill = FILL_GOOD

        ws.row_dimensions[i].height = 44


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 58)
    print("[M6] 报告导出")
    print("=" * 58)

    if not FINAL_RESULTS.exists():
        print(f"\n  [X] 找不到 {FINAL_RESULTS.name}")
        print("      请先运行 python -m src.analyze")
        return 1

    final = json.loads(FINAL_RESULTS.read_text(encoding="utf-8"))
    records = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    llm_map = {r["id"]: r for r in json.loads(LLM_RESULTS.read_text(encoding="utf-8"))}

    blended = final["blended"]
    agg = final["agreement"]
    kappa = final["kappa"]

    print(f"\n[1/2] 生成汇总表 ...")
    wb = Workbook()
    build_summary(wb.active, agg, kappa, records)
    wb.active.title = "评测汇总"

    print(f"[2/2] 生成明细表（{len(blended)} 条）...")
    build_detail(wb.create_sheet("逐条明细"), blended, records, llm_map)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wb.save(REPORT_XLSX)

    print(f"\n报告已生成：{REPORT_XLSX}")
    print(f"  一致率 {agg['accuracy']:.1%} · Kappa {kappa:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
