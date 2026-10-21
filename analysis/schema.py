"""列语义识别与类型转换。

任务数据列名未知，先按同义词打分映射，再按映射结果做数值/日期转换。
映射不中的列原样保留并在结果里列出，绝不静默丢弃。

映射分两轮：完全相等（exact）优先，包含关系（contains）次之；包含匹配若出现
多个候选，会在 mapping_detail 里标为歧义，提醒人工确认，避免把 “Unit Cost”
误当总成本这类错误悄悄带进结论。
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# 同义词表；顺序即优先级。正式数据出现新列名时，只需在这里补一行。
SYNONYMS: dict[str, list[str]] = {
    "order_id": ["订单编号", "订单号", "订单id", "orderid", "order_id", "订单"],
    "date": ["订单日期", "下单日期", "下单时间", "日期", "date", "order_date", "dt"],
    "region": ["地区", "区域", "省份", "大区", "region", "province"],
    "city": ["城市", "city"],
    "category": ["产品类别", "品类", "类别", "产品分类", "category", "category_name"],
    "product": ["产品名称", "商品名称", "产品", "商品", "product", "product_name", "item"],
    "channel": ["销售渠道", "渠道", "channel"],
    "price": ["单价", "价格", "unit_price", "unitprice", "price"],
    "qty": ["数量", "购买数量", "销量", "件数", "quantity", "qty", "count"],
    "amount": ["销售额", "销售金额", "金额", "营收", "revenue", "amount", "sales"],
    "cost": ["成本", "总成本", "cost", "total_cost"],
    "rating": ["客户评分", "评分", "rating", "score", "评价"],
    "returned": ["是否退货", "退货", "returned", "is_return", "return_flag"],
}

NUMERIC_ROLES = ("price", "qty", "amount", "cost", "rating")

# 正式分析至少需要的角色；可在命令行用 --require 覆盖。
DEFAULT_REQUIRED = ("amount",)


def _normalize(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name)).strip().lower()
    return re.sub(r"[\s_\-（）()【】\[\]:：/]+", "", text)


def _semantic_warning(role: str, col: str) -> str | None:
    """列名本身透露出的口径风险，例如把单位成本当总成本。"""
    norm = _normalize(col)
    if role == "cost" and any(k in norm for k in ("unit", "单位", "单价", "per")):
        return "cost_column_looks_like_unit_cost"
    if role == "amount" and "unit" in norm:
        return "amount_column_looks_like_unit_price"
    return None


def map_columns(columns: list[str]) -> tuple[dict[str, str], dict, list[str]]:
    """返回 ({角色: 列名}, 映射细节, 未映射列)。一个列只匹配一个角色。"""
    normalized = {col: _normalize(col) for col in columns}
    used: set[str] = set()
    mapping: dict[str, str] = {}
    detail: dict[str, dict] = {}

    # 第一轮：完全相等
    for role, synonyms in SYNONYMS.items():
        for syn in synonyms:
            target = _normalize(syn)
            for col, norm in normalized.items():
                if col in used or norm != target:
                    continue
                mapping[role] = col
                used.add(col)
                detail[role] = {
                    "column": col,
                    "match": "exact",
                    "candidates": [col],
                    "warning": _semantic_warning(role, col),
                }
                break
            if role in mapping:
                break

    # 第二轮：包含关系（要求长度 >= 2，避免 "id" 之类误伤）
    for role, synonyms in SYNONYMS.items():
        if role in mapping:
            continue
        candidates: list[str] = []
        for syn in synonyms:
            target = _normalize(syn)
            if len(target) < 2:
                continue
            for col, norm in normalized.items():
                if col in used or col in candidates:
                    continue
                if target in norm or (len(norm) >= 2 and norm in target):
                    candidates.append(col)
        if candidates:
            mapping[role] = candidates[0]
            used.add(candidates[0])
            detail[role] = {
                "column": candidates[0],
                "match": "contains",
                "candidates": candidates,
                "ambiguous": len(candidates) > 1,
                "warning": _semantic_warning(role, candidates[0]),
            }

    unmapped = [col for col in columns if col not in used]
    return mapping, detail, unmapped


def to_number(series: pd.Series) -> pd.Series:
    """清洗货币符号、千分位、百分号，括号负数；无法解析为 NaN。"""
    text = (
        series.astype("string")
        .str.strip()
        .str.replace(r"[¥￥$€,，\s]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    text = text.str.replace("%", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def to_datetime(series: pd.Series) -> pd.Series:
    """尽力解析日期；无法解析为 NaT。不猜 dayfirst，交给 pandas 的混合解析。"""
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def coerce(df: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """基于映射生成类型化副本；同时记录每列解析失败数，不改原始 df。"""
    typed = df.copy()
    failures: dict[str, dict] = {}
    for role in NUMERIC_ROLES:
        col = mapping.get(role)
        if not col:
            continue
        raw = df[col]
        parsed = to_number(raw)
        failed = raw.notna() & parsed.isna()
        typed[col] = parsed
        if int(failed.sum()):
            failures[role] = {
                "column": col,
                "non_null": int(raw.notna().sum()),
                "parse_failed": int(failed.sum()),
                "examples": [
                    {"row_index": int(idx), "raw": str(raw.loc[idx])[:40]}
                    for idx in raw[failed].index[:5]
                ],
            }

    date_col = mapping.get("date")
    if date_col:
        raw = df[date_col]
        parsed = to_datetime(raw)
        failed = raw.notna() & parsed.isna()
        typed[date_col] = parsed
        if int(failed.sum()):
            failures["date"] = {
                "column": date_col,
                "non_null": int(raw.notna().sum()),
                "parse_failed": int(failed.sum()),
                "examples": [
                    {"row_index": int(idx), "raw": str(raw.loc[idx])[:40]}
                    for idx in raw[failed].index[:5]
                ],
            }
    return typed, failures
