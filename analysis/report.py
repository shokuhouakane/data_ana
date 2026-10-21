"""结果导出：results.json（机器可读，README 只引用它）+ summary.md + 图。

分组汇总显式记录 total_groups / displayed / truncated，避免 head(20) 造成
“分组和 ≠ 总额”却无标记；路径统一为 POSIX 相对路径。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

GROUP_ROLES = ("region", "city", "category", "product", "channel", "returned")
MISSING_LABEL = "（缺失）"


def _clean(value):
    """把 numpy / NaN 转成 JSON 可序列化类型。"""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _round(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return value
    if out != out:
        return None
    return int(out) if out.is_integer() else round(out, 2)


def _label(value) -> str:
    if value is None or (isinstance(value, float) and value != value) or pd.isna(value):
        return MISSING_LABEL
    return str(value)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_summaries(typed: pd.DataFrame, mapping: dict[str, str]) -> dict:
    amount_col = mapping.get("amount")
    qty_col = mapping.get("qty")
    cost_col = mapping.get("cost")
    date_col = mapping.get("date")
    overall_amount = float(typed[amount_col].sum()) if amount_col else None
    summaries: dict = {}

    for role in GROUP_ROLES:
        col = mapping.get(role)
        if not col or col not in typed.columns:
            continue
        grouped = typed.groupby(col, dropna=False)
        metrics = {c: "sum" for c in (amount_col, qty_col, cost_col) if c}
        table = grouped.agg(metrics)
        table["rows"] = grouped.size()
        if amount_col:
            table = table.sort_values(amount_col, ascending=False)
            group_sum = float(table[amount_col].sum())
        else:
            group_sum = None
        displayed = table.head(20)
        summaries[role] = {
            "column": col,
            "total_groups": int(table.shape[0]),
            "displayed": int(displayed.shape[0]),
            "truncated": bool(table.shape[0] > displayed.shape[0]),
            "group_sum_all_amount": round(group_sum, 2) if group_sum is not None else None,
            "overall_amount": round(overall_amount, 2) if overall_amount is not None else None,
            "reconciles": (
                None if group_sum is None or overall_amount is None
                else abs(group_sum - overall_amount) < 0.01
            ),
            "rows": [
                {"value": _label(idx), **{k: _round(v) for k, v in row.items()}}
                for idx, row in displayed.iterrows()
            ],
        }

    if date_col and amount_col:
        valid = typed[typed[date_col].notna()].copy()
        if len(valid):
            valid["__month"] = valid[date_col].dt.to_period("M").astype(str)
            agg = {amount_col: "sum"}
            if cost_col:
                agg[cost_col] = "sum"
            monthly = valid.groupby("__month").agg(agg)
            monthly["rows"] = valid.groupby("__month").size()
            monthly["first_day"] = valid.groupby("__month")[date_col].min().dt.strftime("%Y-%m-%d")
            monthly["last_day"] = valid.groupby("__month")[date_col].max().dt.strftime("%Y-%m-%d")
            summaries["monthly"] = {
                "column": date_col,
                "rows": [
                    {"month": str(idx), **{k: _round(v) for k, v in row.items()}}
                    for idx, row in monthly.iterrows()
                ],
            }
    return summaries


def write_json(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps(_clean(results), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_summary_md(results: dict, out_dir: Path) -> Path:
    lines = [
        "# 数据体检摘要（由脚本生成，请勿手改）",
        "",
        f"- 数据文件：`{results['meta']['file']}`",
        f"- sha256：`{results['meta']['sha256']}`",
        f"- 行数：{results['quality']['row_count']}，列数：{results['quality']['column_count']}",
        f"- 编码：{results['meta']['encoding']}，分隔符：`{results['meta']['delimiter']}`",
        f"- 语义映射：{json.dumps(results['schema']['mapped'], ensure_ascii=False)}",
        "",
        "## 默认口径 totals",
        "",
        "```json",
        json.dumps(results["policy"]["default_totals"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 体检",
        "",
        "```json",
        json.dumps(results["quality"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 分组概览",
        "",
    ]
    for role, block in results.get("summaries", {}).items():
        lines.append(f"### {role}（{block.get('column')}）")
        lines.append("")
        if "total_groups" in block:
            lines.append(
                f"分组数 {block['total_groups']}，展示 {block['displayed']}，"
                f"截断={block['truncated']}，分组和与总额一致={block['reconciles']}"
            )
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(block["rows"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_figures(typed: pd.DataFrame, mapping: dict[str, str], out_dir: Path) -> list[str]:
    """只画两张通用图；正式结论图在业务分析层另写。"""
    created: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return created

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    amount_col = mapping.get("amount")
    date_col = mapping.get("date")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if amount_col and date_col:
        valid = typed[typed[date_col].notna()]
        if len(valid):
            monthly = valid.groupby(valid[date_col].dt.to_period("M").astype(str))[amount_col].sum()
            fig, ax = plt.subplots(figsize=(9, 4.2))
            ax.bar(monthly.index, monthly.values, color="#4C78A8")
            ax.set_title(f"各月{amount_col}合计（n={len(valid)} 行）")
            ax.set_ylabel(amount_col)
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            path = fig_dir / "monthly_amount.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            created.append(_rel(path))

    cat_col = mapping.get("category") or mapping.get("region")
    if amount_col and cat_col:
        top = (
            typed.groupby(cat_col, dropna=False)[amount_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.barh([_label(x) for x in top.index][::-1], top.values[::-1], color="#59A14F")
        ax.set_title(f"按{cat_col}的{amount_col}前 10（n={len(typed)} 行）")
        ax.set_xlabel(amount_col)
        fig.tight_layout()
        path = fig_dir / "top_categories.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        created.append(_rel(path))

    return created
