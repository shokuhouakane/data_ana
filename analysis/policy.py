"""口径层：把“去重 / 退货 / 金额以谁为准”三种选择落成可复算的 totals。

体检层只报事实，这里才做选择。每种组合都算一遍并写入 results.json，
README 引用时只需写明用了哪一组，评审可按同一脚本复算。
"""

from __future__ import annotations

from itertools import product

import pandas as pd

TRUTHY = {"是", "y", "yes", "true", "1", "退货", "已退货"}

RULE_TEXT = {
    "dedup": {
        "none": "不去重",
        "full_row": "整行完全相同只保留一条",
        "order_id": "同一订单号只保留第一条",
    },
    "returns": {
        "include": "退货行按原始记录计入（负销售额会冲减）",
        "exclude": "剔除退货行",
    },
    "amount": {
        "amount": "以原始销售额为准",
        "price_x_qty": "以 单价×数量 重算，无法重算时回退原始销售额",
    },
}


def return_mask(df: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    col = mapping.get("returned")
    if not col or col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype("string").str.strip().str.lower().isin(TRUTHY)


def _dedup_mask(df: pd.DataFrame, typed: pd.DataFrame, mapping: dict[str, str], rule: str):
    if rule == "none":
        return pd.Series(True, index=df.index)
    if rule == "full_row":
        return ~df.duplicated(keep="first")
    if rule == "order_id":
        col = mapping.get("order_id")
        if not col:
            return None
        ids = typed[col]
        return ~ids.duplicated(keep="first") | ids.isna()
    return None


def _amount_series(typed: pd.DataFrame, mapping: dict[str, str], authority: str):
    amount_col = mapping.get("amount")
    raw = typed[amount_col] if amount_col else pd.Series(float("nan"), index=typed.index)
    if authority == "amount":
        return raw, 0
    price_col, qty_col = mapping.get("price"), mapping.get("qty")
    if not (price_col and qty_col):
        return raw, 0
    calc = typed[price_col] * typed[qty_col]
    fallback = calc.isna() & raw.notna()
    return calc.fillna(raw), int(fallback.sum())


def _totals(df, typed, mapping, mask, returns, authority) -> dict:
    amount, fallback = _amount_series(typed, mapping, authority)
    cost_col, qty_col = mapping.get("cost"), mapping.get("qty")
    return {
        "rows": int(mask.sum()),
        "returns_rows": int((mask & returns).sum()),
        "amount": round(float(amount[mask].sum()), 2),
        "cost": round(float(typed.loc[mask, cost_col].sum()), 2) if cost_col else None,
        "qty": round(float(typed.loc[mask, qty_col].sum()), 2) if qty_col else None,
        "amount_fallback_rows": fallback,
    }


def build_policy_block(df: pd.DataFrame, typed: pd.DataFrame, mapping: dict[str, str]) -> dict:
    returns = return_mask(df, mapping)
    has_returns = bool(mapping.get("returned"))
    has_recalc = bool(mapping.get("price") and mapping.get("qty") and mapping.get("amount"))

    dedup_rules = ["none", "full_row"]
    if mapping.get("order_id"):
        dedup_rules.append("order_id")
    return_rules = ["include", "exclude"] if has_returns else ["include"]
    amount_rules = ["amount"] + (["price_x_qty"] if has_recalc else [])

    combos = []
    for dedup, ret, auth in product(dedup_rules, return_rules, amount_rules):
        base = _dedup_mask(df, typed, mapping, dedup)
        if base is None:
            continue
        mask = base if ret == "include" else (base & ~returns)
        combos.append({
            "dedup": dedup,
            "returns": ret,
            "amount_authority": auth,
            "rules": f"{RULE_TEXT['dedup'][dedup]}；{RULE_TEXT['returns'][ret]}；{RULE_TEXT['amount'][auth]}",
            "totals": _totals(df, typed, mapping, mask, returns, auth),
        })

    # 默认口径：整行去重、退货按原记录、销售额以原值为准
    default = {"dedup": "full_row", "returns": "include", "amount_authority": "amount"}
    chosen = next(
        (c for c in combos if all(c[k] == v for k, v in {
            "dedup": default["dedup"],
            "returns": default["returns"],
            "amount_authority": default["amount_authority"],
        }.items())),
        combos[0] if combos else None,
    )

    return {
        "default": default,
        "default_totals": chosen["totals"] if chosen else None,
        "rule_text": RULE_TEXT,
        "combos": combos,
    }
