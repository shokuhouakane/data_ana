#!/usr/bin/env python
"""业务分析：定位配置失误的关键症结，并评估“新鲜度门禁 + 转人工”策略。

用法：
    python analyze_faults.py --data assistant_query_logs.csv --out results

输出：
    results/business.json      README 引用的全部数字
    results/figures/fail_by_lag.png
    results/figures/fail_by_team_strata.png
    results/figures/cost_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.io_utils import read_table
from analysis.schema import to_number

STALE_MIN = 20          # 数据中失败率在 20 分钟处断层
FAIL_COST_MIN = 90      # 任务书口径：一次故障约 90 分钟
ALERT_COST_MIN = 12     # 任务书口径：一次预警转人工约 12 分钟
THRESHOLDS = [10, 12, 15, 18, 20, 22, 25, 30, 35, 40]


def prepare(path: Path) -> tuple[pd.DataFrame, dict]:
    raw, meta = read_table(path)
    df = pd.DataFrame({
        "query_id": raw["query_id"],
        "query_time": pd.to_datetime(raw["query_time"], errors="coerce", utc=True),
        "kb_synced_at": pd.to_datetime(raw["kb_synced_at"], errors="coerce", utc=True),
        "team": raw["deployment_team"].astype("string").str.strip(),
        "category": raw["knowledge_category"].astype("string").str.strip(),
        "match_margin": to_number(raw["match_margin"]),
        "inference_ms": to_number(raw["inference_duration_ms"]),
        "is_success": to_number(raw["is_success"]),
    })
    df["lag_min"] = (df["query_time"] - df["kb_synced_at"]).dt.total_seconds() / 60
    df["stale"] = df["lag_min"] >= STALE_MIN
    df["fail"] = df["is_success"].map({1.0: False, 0.0: True})
    df["local_hour"] = df["query_time"].dt.tz_convert("Asia/Shanghai").dt.hour
    df["local_date"] = df["query_time"].dt.tz_convert("Asia/Shanghai").dt.date

    # query_id 是任务书定义的单据唯一流水号；重复行必须先声明再剔除
    raw_rows = int(len(df))
    dup_mask = df.duplicated(subset=["query_id"], keep="first")
    duplicate_rows = int(dup_mask.sum())
    before_labeled = df[df["is_success"].notna()]
    before_fail = int((before_labeled["is_success"] == 0).sum())
    before_stale_fail = int(((before_labeled["is_success"] == 0) & before_labeled["stale"]).sum())
    meta["before_dedup"] = {
        "rows": raw_rows,
        "labeled": int(len(before_labeled)),
        "failures": before_fail,
        "fail_rate": round(before_fail / len(before_labeled), 4) if len(before_labeled) else None,
        "stale_share_of_failures": round(before_stale_fail / before_fail, 4) if before_fail else None,
    }
    fresh_rows = int((~df["stale"]).sum())
    stale_rows = int(df["stale"].sum())
    meta["duplicates_by_stale"] = {
        "fresh": int((dup_mask & ~df["stale"]).sum()),
        "stale": int((dup_mask & df["stale"]).sum()),
        "fresh_share": round(int((dup_mask & ~df["stale"]).sum()) / fresh_rows, 4) if fresh_rows else None,
        "stale_share": round(int((dup_mask & df["stale"]).sum()) / stale_rows, 4) if stale_rows else None,
    }
    df = df.drop_duplicates(subset=["query_id"], keep="first").copy()
    meta["raw_rows"] = raw_rows
    meta["duplicate_rows_removed"] = duplicate_rows
    meta["unique_query_ids"] = int(df["query_id"].nunique())
    meta["dedup_rule"] = "同一 query_id 只保留第一条"
    return df, meta


def rates(df: pd.DataFrame, mask: pd.Series) -> dict:
    part = df[mask]
    return {
        "queries": int(len(part)),
        "failures": int(part["fail"].sum()),
        "fail_rate": round(float(part["fail"].mean()), 4) if len(part) else None,
    }


def threshold_table(all_df: pd.DataFrame) -> list[dict]:
    """预警发生在查询时，未回传结果的查询同样会触发；故障只统计有回传的。"""
    df = all_df[all_df["is_success"].notna()]
    total_fails = int(df["fail"].sum())
    baseline = total_fails * FAIL_COST_MIN
    rows = []
    for t in THRESHOLDS:
        alert_all = all_df["lag_min"] >= t
        alerts = int(alert_all.sum())
        caught = int((alert_all & all_df["fail"].eq(True)).sum())
        remaining = total_fails - caught
        cost = alerts * ALERT_COST_MIN + remaining * FAIL_COST_MIN
        rows.append({
            "threshold_min": t,
            "alerts": alerts,
            "alert_share": round(alerts / len(all_df), 4),
            "failures_caught": caught,
            "failures_total": total_fails,
            "recall": round(caught / total_fails, 4),
            "remaining_failures": remaining,
            "cost_minutes": int(cost),
            "baseline_cost_minutes": int(baseline),
            "saved_minutes": int(baseline - cost),
        })
    return rows


def strategy_sensitivity(all_df: pd.DataFrame, threshold: int) -> dict:
    df = all_df[all_df["is_success"].notna()]
    total_fails = int(df["fail"].sum())
    alert_all = all_df["lag_min"] >= threshold
    alerts_all = int(alert_all.sum())
    alerts_labeled = int((df["lag_min"] >= threshold).sum())
    caught = int((alert_all & all_df["fail"].eq(True)).sum())
    out = {}
    for prevent in (1.0, 0.8, 0.6):
        remaining = total_fails - caught * prevent
        cost = alerts_all * ALERT_COST_MIN + remaining * FAIL_COST_MIN
        out[f"assume_prevented_{int(prevent * 100)}pct"] = {
            "cost_minutes": round(cost, 1),
            "saved_minutes": round(total_fails * FAIL_COST_MIN - cost, 1),
            "saved_pct": round((total_fails * FAIL_COST_MIN - cost) / (total_fails * FAIL_COST_MIN) * 100, 1),
        }
    return {
        "threshold_min": threshold,
        "alerts_labeled": alerts_labeled,
        "alerts_all_queries": alerts_all,
        "failures_caught": caught,
        "failures_total": total_fails,
        "per_team_per_day_alerts": round(
            alerts_all
            / all_df["team"].nunique()
            / all_df["query_time"].dt.tz_convert("Asia/Shanghai").dt.date.nunique(),
            1,
        ),
        "sensitivity": out,
    }


def _standardize(labeled: pd.DataFrame, key: str) -> dict:
    """直接标准化：消除团队/类别构成差异后，过期与新鲜的故障率。"""
    table = labeled.groupby([key, "stale"])["fail"].mean().unstack()
    both = table.dropna()
    w_fresh = labeled[~labeled["stale"]].groupby(key).size().reindex(both.index)
    w_stale = labeled[labeled["stale"]].groupby(key).size().reindex(both.index)
    return {
        "strata": int(len(both)),
        "stale_rate_standardized_to_fresh": round(float((both[True] * w_fresh / w_fresh.sum()).sum()), 4),
        "fresh_rate_standardized_to_stale": round(float((both[False] * w_stale / w_stale.sum()).sum()), 4),
    }


def _logistic_stale_or(labeled: pd.DataFrame) -> dict:
    """逻辑回归：故障 ~ 过期 + margin + 推理 + 团队 + 类别 + 小时。"""
    cols = labeled[["stale", "match_margin", "inference_ms", "team", "category", "local_hour"]].copy()
    X = pd.get_dummies(cols, columns=["team", "category", "local_hour"], drop_first=True, dtype=float)
    X.insert(0, "const", 1.0)
    names = list(X.columns)
    Xv = X.to_numpy(dtype=float)
    y = labeled["fail"].astype(float).to_numpy()
    beta = np.zeros(Xv.shape[1])
    penalty = 1e-3 * np.eye(Xv.shape[1])
    for _ in range(200):
        eta = np.clip(Xv @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-8, None)
        hessian = Xv.T @ (Xv * w[:, None]) + penalty
        gradient = Xv.T @ (y - p)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = np.clip(Xv @ beta, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    w = np.clip(p * (1 - p), 1e-8, None)
    hessian = Xv.T @ (Xv * w[:, None]) + penalty
    se = np.sqrt(np.diag(np.linalg.inv(hessian)))
    i = names.index("stale")
    return {
        "model": "fail ~ stale + match_margin + inference_ms + team + category + local_hour",
        "n": int(len(y)),
        "parameters": int(Xv.shape[1]),
        "stale_odds_ratio": round(float(np.exp(beta[i])), 2),
        "stale_ci95": [
            round(float(np.exp(beta[i] - 1.96 * se[i])), 2),
            round(float(np.exp(beta[i] + 1.96 * se[i])), 2),
        ],
    }


def build_robustness(df: pd.DataFrame, labeled: pd.DataFrame, meta: dict) -> dict:
    missing = {
        "fresh_pct": round(float(df.loc[~df["stale"], "is_success"].isna().mean()), 4),
        "stale_pct": round(float(df.loc[df["stale"], "is_success"].isna().mean()), 4),
    }

    margin = {}
    for s, name in ((False, "fresh"), (True, "stale")):
        part = labeled.loc[labeled["stale"] == s, "match_margin"]
        margin[name] = {
            "mean": round(float(part.mean()), 2),
            "median": round(float(part.median()), 2),
            "p25": round(float(part.quantile(0.25)), 2),
            "p75": round(float(part.quantile(0.75)), 2),
        }

    lab = labeled.copy()
    lab["ibin"] = pd.cut(lab["inference_ms"], [-1, 42, 63, 200], labels=["<=42", "43-63", ">63"])
    inference = []
    for b in ["<=42", "43-63", ">63"]:
        row = {"bin": b}
        for s, name in ((False, "fresh"), (True, "stale")):
            part = lab[(lab["ibin"] == b) & (lab["stale"] == s)]
            row[f"{name}_n"] = int(len(part))
            row[f"{name}_fail_rate"] = round(float(part["fail"].mean()), 4) if len(part) else None
        inference.append(row)

    hours = []
    for hour, frame in lab.groupby("local_hour"):
        fresh, stale = frame[~frame["stale"]], frame[frame["stale"]]
        hours.append({
            "hour": int(hour),
            "fresh_n": int(len(fresh)),
            "fresh_fail_rate": round(float(fresh["fail"].mean()), 4) if len(fresh) else None,
            "stale_n": int(len(stale)),
            "stale_fail_rate": round(float(stale["fail"].mean()), 4) if len(stale) else None,
        })

    volume = (
        df.groupby(["team", "local_date", "local_hour"]).size().rename("volume").reset_index()
    )
    lab = lab.merge(volume, on=["team", "local_date", "local_hour"], how="left")
    lab["volume_bin"] = pd.qcut(lab["volume"], 4, duplicates="drop")
    volume_rows = [
        {
            "volume_bin": str(k),
            "queries": int(len(part)),
            "mean_volume": round(float(part["volume"].mean()), 2),
            "fail_rate": round(float(part["fail"].mean()), 4),
            "stale_share": round(float(part["stale"].mean()), 4),
        }
        for k, part in lab.groupby("volume_bin", observed=True)
    ]

    return {
        "missingness": missing,
        "duplicates": meta.get("duplicates_by_stale"),
        "margin_distribution": margin,
        "inference_strata": inference,
        "hour_strata": hours,
        "volume": {
            "quartiles": volume_rows,
            "corr_volume_fail": round(float(lab["volume"].corr(lab["fail"])), 4),
            "corr_volume_lag": round(float(lab["volume"].corr(lab["lag_min"])), 4),
        },
        "standardization": {
            "by_team": _standardize(labeled, "team"),
            "by_category": _standardize(labeled, "category"),
        },
        "logistic": _logistic_stale_or(labeled),
    }


def build_results(df: pd.DataFrame, meta: dict) -> dict:
    labeled = df[df["is_success"].notna()].copy()
    unlabeled = df[df["is_success"].isna()]
    fresh = labeled[~labeled["stale"]]
    stale = labeled[labeled["stale"]]

    fail_total = int(labeled["fail"].sum())
    stale_fail = int(stale["fail"].sum())

    # 混杂检验：分层内看团队/类别/裕度/推理时长
    team = []
    for name, part in labeled.groupby("team"):
        team.append({
            "team": str(name),
            "queries": int(len(part)),
            "stale_share": round(float(part["stale"].mean()), 4),
            "median_lag_min": round(float(part["lag_min"].median()), 1),
            "median_lag_stale_min": (
                round(float(part.loc[part["stale"], "lag_min"].median()), 1)
                if part["stale"].any() else None
            ),
            **{f"fail_rate_{'stale' if s else 'fresh'}": round(float(part.loc[part["stale"] == s, "fail"].mean()), 4)
               for s in (False, True) if len(part.loc[part["stale"] == s])},
            **{f"n_{'stale' if s else 'fresh'}": int((part["stale"] == s).sum()) for s in (False, True)},
        })
    category = []
    for name, part in labeled.groupby("category"):
        category.append({
            "category": str(name),
            "queries": int(len(part)),
            "stale_share": round(float(part["stale"].mean()), 4),
            "fail_rate_fresh": round(float(part.loc[~part["stale"], "fail"].mean()), 4),
            "fail_rate_stale": round(float(part.loc[part["stale"], "fail"].mean()), 4),
        })

    margin_bins = pd.cut(fresh["match_margin"], [-1, 4, 8, 12, 16, 30],
                         labels=["<=4", "5-8", "9-12", "13-16", ">16"])
    margin = [
        {"bin": str(k), "queries": int(v), "fail_rate": round(float(fresh.loc[margin_bins == k, "fail"].mean()), 4)}
        for k, v in margin_bins.value_counts().sort_index().items()
    ]
    infer_bins = pd.cut(fresh["inference_ms"], [-1, 42, 63, 200], labels=["<=42", "43-63", ">63"])
    infer = [
        {"bin": str(k), "queries": int(v), "fail_rate": round(float(fresh.loc[infer_bins == k, "fail"].mean()), 4)}
        for k, v in infer_bins.value_counts().sort_index().items()
    ]

    thresholds = threshold_table(df)
    best = min(thresholds, key=lambda r: r["cost_minutes"])

    lag_values = sorted(labeled["lag_min"].round(0).astype(int).unique())
    gap = [x for x in range(int(labeled["lag_min"].min()), int(labeled["lag_min"].max()) + 1) if x not in lag_values]

    minute = labeled.assign(m=labeled["lag_min"].round().astype(int))
    exact_minutes = []
    for m in range(15, 26):
        part = minute[minute["m"] == m]
        exact_minutes.append({
            "minute": m,
            "queries": int(len(part)),
            "failures": int(part["fail"].sum()),
            "fail_rate": round(float(part["fail"].mean()), 4) if len(part) else None,
        })
    correlations = {
        "lag_min": round(float(labeled["lag_min"].corr(labeled["fail"])), 4),
        "match_margin": round(float(labeled["match_margin"].corr(labeled["fail"])), 4),
        "inference_ms": round(float(labeled["inference_ms"].corr(labeled["fail"])), 4),
    }
    unlabeled_split = {
        "fresh": int((unlabeled["lag_min"] < STALE_MIN).sum()),
        "stale": int((unlabeled["lag_min"] >= STALE_MIN).sum()),
    }

    local = df["query_time"].dt.tz_convert("Asia/Shanghai")
    return {
        "meta": {
            "file": Path(meta["file"]).name,
            "rows": int(len(df)),
            "raw_rows": meta.get("raw_rows"),
            "duplicate_rows_removed": meta.get("duplicate_rows_removed"),
            "dedup_rule": meta.get("dedup_rule"),
            "before_dedup": meta.get("before_dedup"),
            "labeled": int(len(labeled)),
            "unlabeled": int(len(unlabeled)),
            "days": int(local.dt.date.nunique()),
            "start": local.min().strftime("%Y-%m-%d %H:%M"),
            "end": local.max().strftime("%Y-%m-%d %H:%M"),
            "timezone": "Asia/Shanghai",
            "stale_threshold_min": STALE_MIN,
            "fail_cost_min": FAIL_COST_MIN,
            "alert_cost_min": ALERT_COST_MIN,
        },
        "overall": {
            "failures": fail_total,
            "fail_rate": round(fail_total / len(labeled), 4),
            "unlabeled_share": round(len(unlabeled) / len(df), 4),
        },
        "lag": {
            "fresh": rates(labeled, ~labeled["stale"]),
            "stale": rates(labeled, labeled["stale"]),
            "stale_share_of_queries": round(len(stale) / len(labeled), 4),
            "stale_share_of_failures": round(stale_fail / fail_total, 4),
            "gap_minutes_without_queries": gap,
            "lag_min": round(float(labeled["lag_min"].min()), 2),
            "lag_max": round(float(labeled["lag_min"].max()), 2),
            "exact_minutes_15_25": exact_minutes,
            "unlabeled_split": unlabeled_split,
        },
        "correlations": correlations,
        "confounders": {
            "median_inference_fresh": float(fresh["inference_ms"].median()),
            "median_inference_stale": float(stale["inference_ms"].median()),
            "margin_within_fresh": margin,
            "inference_within_fresh": infer,
            "team": team,
            "category": category,
        },
        "strategy": {
            "thresholds": thresholds,
            "best": best,
            "recommended": strategy_sensitivity(df, STALE_MIN),
        },
        "robustness": build_robustness(df, labeled, meta),
    }


def make_figures(df: pd.DataFrame, results: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labeled = df[df["is_success"].notna()]
    created = []

    # 图 1：失败率随同步滞后的变化（20 分钟断层）
    grouped = labeled.assign(lag=labeled["lag_min"].round().astype(int)).groupby("lag")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(grouped["fail"].mean().index, grouped["fail"].mean().values, marker="o", color="#C44E52")
    ax.axvline(STALE_MIN, color="#333333", linestyle="--", linewidth=1)
    ax.text(STALE_MIN + 1, 0.45, "20 分钟", fontsize=9)
    ax.set_xlabel("知识库同步滞后（分钟）")
    ax.set_ylabel("故障率")
    ax.set_title(f"故障率在同步滞后 20 分钟处断层（n={len(labeled)} 条有回传）")
    fig.tight_layout()
    path = fig_dir / "fail_by_lag.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path.as_posix())

    # 图 2：分层后团队差异（新鲜 vs 过期）
    teams = [t for t in results["confounders"]["team"]]
    names = [t["team"] for t in teams]
    fresh_rates = [t.get("fail_rate_fresh", 0) for t in teams]
    stale_rates = [t.get("fail_rate_stale", 0) for t in teams]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar([i - 0.2 for i in x], fresh_rates, width=0.4, label="同步滞后 <20 分钟", color="#4C78A8")
    ax.bar([i + 0.2 for i in x], stale_rates, width=0.4, label="同步滞后 ≥20 分钟", color="#C44E52")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("故障率")
    ax.set_title("按团队分层：差异主要来自知识库是否过期")
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "fail_by_team_strata.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path.as_posix())

    # 图 3：阈值—成本曲线
    rows = results["strategy"]["thresholds"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot([r["threshold_min"] for r in rows], [r["cost_minutes"] for r in rows], marker="o", color="#59A14F")
    ax.axhline(results["strategy"]["best"]["baseline_cost_minutes"], color="#888888", linestyle=":", label="不干预基线")
    ax.set_xlabel("预警阈值：同步滞后 ≥ t 分钟转人工")
    ax.set_ylabel("总工时（分钟）")
    ax.set_title(f"阈值 20 分钟时成本最低（{results['strategy']['best']['cost_minutes']} 分钟）")
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "cost_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path.as_posix())
    return created


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="assistant_query_logs.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    df, meta = prepare(Path(args.data))
    results = build_results(df, meta)
    results["figures"] = make_figures(df, results, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "business.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "overall": results["overall"],
        "lag": results["lag"],
        "best": results["strategy"]["best"],
        "recommended": results["strategy"]["recommended"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
