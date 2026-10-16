"""
[M0] 数据加载层

职责（只干这一件事）：把原始 xlsx 读进来，清洗成统一的记录对象，写出 records.json。

为什么不直接读 Excel：
    原始表是给人看的 —— 有标题行、有合并单元格的统计区、有说明 sheet。
    直接读会踩三个坑（见下方注释）。清洗一次、固化成 JSON，后面所有模块都读这个，
    规则引擎和大模型评审就不必各自去处理格式问题。

运行方式（在项目根目录，即 ai-cs-evaluator/ 下执行）：
    .venv/Scripts/python -m src.loader
"""

import json
import sys

import pandas as pd

from src.config import (
    COLUMN_MAP,
    HEADER_ROW,
    PROCESSED_DIR,
    RAW_XLSX,
    RECORDS_JSON,
    SHEET_NAME,
    VALID_LABELS,
    VALID_REASONS,
)

# 期望的总轮数。写死成一个常量，是为了让校验能发现「表被误改」这类问题。
EXPECTED_TOTAL = 30


def load_raw(path=RAW_XLSX) -> pd.DataFrame:
    """把原始 sheet 读成 DataFrame。

    坑 1：这个工作簿有三个 sheet，默认读第一个会读到「统计」区，必须显式指定 sheet 名。
    坑 2：列名在第 3 行而不是第 1 行，pandas 的 header 是 0 起算，所以传 HEADER_ROW=2。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"找不到原始数据表：{path}\n"
            f"请把「AI客服应答实测记录表.xlsx」放到 data/raw/ 并重命名为「客服实测记录.xlsx」。"
        )

    df = pd.read_excel(path, sheet_name=SHEET_NAME, header=HEADER_ROW)

    # 列名统一去空格，避免「序号 」这种看不见的差异导致后面匹配失败
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """清洗：挑列 → 改统一字段名 → 丢空行 → 修类型 → 去首尾空白。"""
    # --- 1. 确认需要的列都在，缺列就直接报错，不要带着残缺数据往下走 ---
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(
            f"原始表缺少这些列：{missing}\n"
            f"实际读到的列：{list(df.columns)}\n"
            f"如果表结构变了，请同步修改 src/config.py 里的 COLUMN_MAP。"
        )

    # --- 2. 只保留需要的列，并把中文列名换成英文字段名 ---
    df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)

    # --- 3. 丢掉整行都是空的行（Excel 底部常有一堆看不见的空白行）---
    df = df.dropna(how="all")

    # --- 4. 序号列修正类型 ---
    # 坑 3：Excel 里的整数被 pandas 读进来会变成 1.0 这种浮点数。
    # 用 to_numeric + coerce，非数字（比如结尾的说明文字行）会变成 NaN，随后被丢掉。
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)

    # --- 5. 文本列去首尾空白，空值统一成空字符串 ---
    text_cols = ["scene", "question", "answer", "human_label", "human_reason", "human_note"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame → 统一的记录对象列表。

    这是整个项目的核心数据结构，后面所有模块都依赖它的字段名，定下来就不要改。
    其中 machine 字段先留空，等评测跑完再回填机器评分。
    """
    records = []
    for row in df.to_dict(orient="records"):
        records.append(
            {
                "id": int(row["id"]),
                "scene": row["scene"],
                "question": row["question"],
                "answer": row["answer"],
                "human_label": row["human_label"],      # 人工判定：答对 / 答偏
                "human_reason": row["human_reason"],    # 人工错因
                "human_note": row["human_note"],        # 人工的判断说明与改法
                "machine": None,                        # 机器评分，评测后回填
            }
        )
    return records


def validate(records: list[dict], expected_total: int = EXPECTED_TOTAL) -> tuple[list[str], dict]:
    """进管道前先自检。

    这一步是测试思维的直接体现：数据不验就用，后面出了问题根本不知道是哪一环坏的。
    返回 (错误列表, 统计摘要)。错误列表为空才算通过。
    """
    errors: list[str] = []

    # --- 数量校验 ---
    if len(records) != expected_total:
        errors.append(f"记录数不对：期望 {expected_total} 条，实际 {len(records)} 条")

    # --- 必填字段校验 ---
    required = ["id", "scene", "question", "answer", "human_label"]
    for r in records:
        for field in required:
            if not r.get(field):
                errors.append(f"第 {r.get('id')} 条缺字段：{field}")

    # --- 序号唯一且连续 ---
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        errors.append("序号有重复")
    if ids and ids != list(range(1, len(ids) + 1)):
        errors.append("序号不是从 1 开始的连续整数")

    # --- 取值合法性 ---
    for r in records:
        if r["human_label"] not in VALID_LABELS:
            errors.append(f"第 {r['id']} 条的结果取值非法：{r['human_label']!r}")
        if r["human_reason"] not in VALID_REASONS:
            errors.append(f"第 {r['id']} 条的错因取值非法：{r['human_reason']!r}")

    # --- 统计摘要 ---
    total = len(records)
    n_bad = sum(1 for r in records if r["human_label"] == "答偏")
    n_good = total - n_bad

    scene_dist: dict[str, int] = {}
    reason_dist: dict[str, int] = {}
    for r in records:
        scene_dist[r["scene"]] = scene_dist.get(r["scene"], 0) + 1
        if r["human_label"] == "答偏":
            reason_dist[r["human_reason"]] = reason_dist.get(r["human_reason"], 0) + 1

    stats = {
        "total": total,
        "good": n_good,
        "bad": n_bad,
        "bad_rate": round(n_bad / total, 4) if total else 0,
        "scene_dist": scene_dist,
        "reason_dist": reason_dist,
    }
    return errors, stats


def save(records: list[dict], path=RECORDS_JSON) -> None:
    """写出 JSON。ensure_ascii=False 才能让中文正常显示而不是 \\uXXXX。"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    # Windows 控制台默认用 GBK，直接 print 中文可能乱码，这里强制走 UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 58)
    print("[M0] 数据加载层")
    print("=" * 58)

    print(f"\n[1/3] 读取原始表 {RAW_XLSX.name} ...")
    raw = load_raw()
    print(f"      读到 {len(raw)} 行 × {len(raw.columns)} 列（含底部空行）")

    print("\n[2/3] 清洗并校验 ...")
    df = clean(raw)
    records = to_records(df)
    errors, stats = validate(records)

    print(f"      清洗后有效记录：{stats['total']} 条")
    print(f"      人工判定：答对 {stats['good']} / 答偏 {stats['bad']}"
          f"（答偏率 {stats['bad_rate']:.1%}）")
    print(f"      场景分布：{stats['scene_dist']}")
    print(f"      答偏错因：{stats['reason_dist']}")

    if errors:
        print("\n  [X] 校验未通过：")
        for e in errors:
            print(f"      - {e}")
        return 1

    print("      [OK] 校验通过")

    print(f"\n[3/3] 写出 {RECORDS_JSON.relative_to(RECORDS_JSON.parent.parent.parent)} ...")
    save(records)
    print(f"      完成，共 {len(records)} 条")
    print("\n下一步：DAY 2 写规则评分引擎 src/rules.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
