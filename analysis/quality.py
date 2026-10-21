"""数据体检：每个异常都报“影响多少钱”，而不是只报行数。

所有检查只描述事实，不做静默剔除；剔除规则属于口径层（policy.py），
必须在 README 声明。异常示例带订单号，方便评审回查原始行。
"""

from __future__ import annotations

import math

import pandas as pd

from analysis.policy import TRUTHY, return_mask


def _f(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else round(out, 2)


def _i(value) -> int:
    return int(value)


def _sum(series: pd.Series) -> float | None:
    return _f(series.sum()) if len(series) else None


def _amount_of(df: pd.DataFrame, mask: pd.Series, amount_col: str | None) -> float | None:
    if not amount_col or amount_col not in df.columns:
        return None
    return _sum(df.loc[mask, amount_col])


def _row_ref(typed: pd.DataFrame, mapping: dict[str, str], idx) -> dict:
    ref = {"row_index": int(idx)}
    id_col = mapping.get("order_id")
    if id_col:
        ref["order_id"] = str(typed.loc[idx, id_col])
    date_col = mapping.get("date")
    if date_col and pd.notna(typed.loc[idx, date_col]):
        ref["date"] = typed.loc[idx, date_col].strftime("%Y-%m-%d")
    return ref


def run_checks(df: pd.DataFrame, typed: pd.DataFrame, mapping: dict[str, str]) -> dict:
    amount_col = mapping.get("amount")
    cost_col = mapping.get("cost")
    qty_col = mapping.get("qty")
    price_col = mapping.get("price")
    id_col = mapping.get("order_id")
    date_col = mapping.get("date")
    returned_col = mapping.get("returned")

    n = len(df)
    out: dict = {"row_count": n, "column_count": int(df.shape[1])}

    # ---------- 重复 ----------
    dup_extra = df.duplicated(keep="first")
    dup_report = {
        "full_row_duplicate_extra_rows": _i(dup_extra.sum()),
        "full_row_duplicate_amount_impact": _amount_of(typed, dup_extra, amount_col),
    }
    if id_col:
        ids = typed[id_col]
        dup_id_mask = ids.notna() & ids.duplicated(keep=False)
        extra = int(dup_id_mask.sum() - ids[dup_id_mask].nunique()) if dup_id_mask.any() else 0
        dup_report.update({
            "order_id": id_col,
            "order_id_missing": _i(ids.isna().sum()),
            "order_id_duplicated_extra_rows": extra,
            "order_id_duplicate_amount_impact": None,
        })
        if extra > 0:
            seen = set()
            extra_mask = pd.Series(False, index=df.index)
            for idx, value in ids.items():
                if pd.isna(value):
                    continue
                if value in seen:
                    extra_mask.loc[idx] = True
                else:
                    seen.add(value)
            dup_report["order_id_duplicate_amount_impact"] = _amount_of(typed, extra_mask, amount_col)
    out["duplicates"] = dup_report

    # ---------- 缺失 ----------
    missing = {}
    for col in df.columns:
        miss = df[col].isna()
        count = int(miss.sum())
        if count == 0:
            continue
        missing[col] = {
            "count": count,
            "pct": round(count / n * 100, 2) if n else None,
            "amount_in_affected_rows": _amount_of(typed, miss, amount_col) if col != amount_col else None,
        }
    out["missing"] = missing

    # ---------- 金额一致性 ----------
    if price_col and qty_col and amount_col:
        price = typed[price_col]
        qty = typed[qty_col]
        amount = typed[amount_col]
        expected = price * qty
        comparable = expected.notna() & amount.notna()
        diff = (amount - expected).abs()
        mismatch = comparable & (diff > 0.01)
        out["amount_consistency"] = {
            "price_col": price_col,
            "qty_col": qty_col,
            "amount_col": amount_col,
            "comparable_rows": _i(comparable.sum()),
            "non_comparable_rows": _i((~comparable).sum()),
            "mismatch_rows": _i(mismatch.sum()),
            "mismatch_abs_diff_total": _f(diff[mismatch].sum()),
            "mismatch_amount_total": _sum(amount[mismatch]),
            "mismatch_expected_total": _sum(expected[mismatch]),
            "examples": [
                {
                    **_row_ref(typed, mapping, idx),
                    "price": _f(price.loc[idx]),
                    "qty": _f(qty.loc[idx]),
                    "amount": _f(amount.loc[idx]),
                    "expected": _f(expected.loc[idx]),
                }
                for idx in amount[mismatch].index[:5]
            ],
        }

    # ---------- 负数量 / 退货 ----------
    if qty_col:
        neg = typed[qty_col] < 0
        out["negative_quantity"] = {
            "rows": _i(neg.sum()),
            "amount_total": _sum(typed.loc[neg, amount_col]) if amount_col else None,
            "cost_total": _sum(typed.loc[neg, cost_col]) if cost_col else None,
            "qty_total": _sum(typed.loc[neg, qty_col]),
            "examples": [_row_ref(typed, mapping, idx) for idx in typed[neg].index[:5]],
        }

    if returned_col:
        raw = df[returned_col]
        is_return = return_mask(df, mapping)
        falsey = {"否", "n", "no", "false", "0", "无", "未退货"}
        normalized = raw.dropna().astype("string").str.strip().str.lower()
        unmatched = sorted(set(normalized) - TRUTHY - falsey)
        non_null = int(raw.notna().sum())
        warning = None
        if non_null and int(is_return.sum()) == 0:
            warning = "退货列非空但没有命中任何“是”类值；退货口径会退化为无退货，请核对取值。"
        out["returns"] = {
            "column": returned_col,
            "value_counts": {str(k): _i(v) for k, v in raw.value_counts(dropna=False).items()},
            "return_rows": _i(is_return.sum()),
            "return_amount_total": _sum(typed.loc[is_return, amount_col]) if amount_col else None,
            "return_cost_total": _sum(typed.loc[is_return, cost_col]) if cost_col else None,
            "truthy_values": sorted(TRUTHY),
            "unmatched_values": unmatched,
            "warning": warning,
        }

    # ---------- 成本语义 ----------
    if cost_col and qty_col and price_col:
        frame = typed[[cost_col, qty_col, price_col]].dropna()
        frame = frame[frame[qty_col] > 0]
        if len(frame):
            cost_med = float(frame[cost_col].median())
            price_med = float(frame[price_col].median())
            unit_cost_med = float((frame[cost_col] / frame[qty_col]).median())
            line_total_med = float((frame[price_col] * frame[qty_col]).median())
            ratio = round(cost_med / price_med, 3) if price_med else None
            # 仅凭数值无法可靠区分“总成本”与“单位成本”（qty 恒为 1 时两者同形），
            # 因此数值只做诊断；列名带 unit/单位/单价 才判为高风险，其余标记“需确认”。
            name = str(cost_col).lower()
            name_hint = any(k in name for k in ("unit", "单位", "单价", "per"))
            out["cost_semantics"] = {
                "cost_col": cost_col,
                "median_cost_per_row": round(cost_med, 2),
                "median_price": round(price_med, 2),
                "median_cost_per_unit": round(unit_cost_med, 2),
                "median_price_times_qty": round(line_total_med, 2),
                "ratio_cost_to_price": ratio,
                "likely_unit_cost": True if name_hint else None,
                "margin_requires_confirmation": True,
                "warning": (
                    "成本列名疑似单位成本；若属实，汇总前必须乘以数量。"
                    if name_hint
                    else "成本列语义无法仅凭数值判定（总成本/单位成本）；未确认前不得计算毛利。"
                ),
            }

    # ---------- 日期 ----------
    if date_col:
        raw = df[date_col]
        dates = typed[date_col]
        valid = dates.dropna()
        invalid = raw.notna() & dates.isna()
        monthly = {}
        if len(valid):
            periods = valid.dt.to_period("M").astype(str)
            monthly = {str(k): _i(v) for k, v in periods.value_counts().sort_index().items()}
        out["dates"] = {
            "column": date_col,
            "missing": _i(raw.isna().sum()),
            "invalid": _i(invalid.sum()),
            "invalid_examples": [
                {"row_index": int(idx), "raw": str(raw.loc[idx])[:40]}
                for idx in raw[invalid].index[:5]
            ],
            "min": valid.min().strftime("%Y-%m-%d") if len(valid) else None,
            "max": valid.max().strftime("%Y-%m-%d") if len(valid) else None,
            "rows_by_month": monthly,
        }

    # ---------- 原始加总 ----------
    out["totals"] = {
        "amount_sum_raw": _sum(typed[amount_col]) if amount_col else None,
        "cost_sum_raw": _sum(typed[cost_col]) if cost_col else None,
        "qty_sum_raw": _sum(typed[qty_col]) if qty_col else None,
    }
    return out
