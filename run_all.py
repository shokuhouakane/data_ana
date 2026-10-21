#!/usr/bin/env python
"""一键复现入口。

用法：
    python run_all.py --data tests/fixtures/synthetic_orders.csv --out results

    # 正式数据若没有“销售额”列，可显式调整必需角色
    python run_all.py --data data/xxx.csv --require qty

输出：
    results/results.json   机器可读结果（README 中的数字只引用这里）
    results/summary.md     体检摘要
    results/figures/*.png  通用概览图（正式结论图另写）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from analysis.io_utils import read_table
from analysis.policy import apply_policy, build_policy_block
from analysis.quality import run_checks
from analysis.report import build_summaries, make_figures, write_json, write_summary_md
from analysis.schema import DEFAULT_REQUIRED, coerce, map_columns


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="通用数据分析流水线")
    parser.add_argument("--data", required=True, help="输入 CSV 路径")
    parser.add_argument("--out", default="results", help="输出目录")
    parser.add_argument(
        "--require",
        default=",".join(DEFAULT_REQUIRED),
        help="必需语义角色，逗号分隔（默认 amount）",
    )
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[FAIL] 找不到数据文件：{data_path}", file=sys.stderr)
        return 1
    out_dir = Path(args.out)

    df, read_meta = read_table(data_path)
    mapping, mapping_detail, unmapped = map_columns(list(df.columns))

    # 必需角色校验：缺了就非零退出，避免静默生成一份没有核心数字的结果
    required = [r.strip() for r in args.require.split(",") if r.strip()]
    missing_roles = [r for r in required if r not in mapping]
    if missing_roles:
        print(f"[FAIL] 缺少必需语义角色：{missing_roles}", file=sys.stderr)
        print(f"[INFO] 已识别：{json.dumps(mapping, ensure_ascii=False)}", file=sys.stderr)
        print(f"[INFO] 未映射列：{unmapped}", file=sys.stderr)
        print("[HINT] 用 --require 调整必需要求，或在 analysis/schema.py 的同义词表补充列名。", file=sys.stderr)
        return 2

    typed, coercion_failures = coerce(df, mapping)
    quality = run_checks(df, typed, mapping)
    policy = build_policy_block(df, typed, mapping)

    # 摘要与图按默认口径过滤后的数据生成，保证与 default_totals 同源
    default = policy["default"]
    mask = apply_policy(df, typed, mapping, default["dedup"], default["returns"])
    typed_default = typed[mask]
    summaries = build_summaries(typed_default, mapping, policy_label=default)
    policy["summary_policy_applied"] = True
    policy["summary_rows"] = int(mask.sum())

    ambiguous = {
        role: info for role, info in mapping_detail.items()
        if info.get("ambiguous") or info.get("warning")
    }

    results = {
        "meta": {
            "file": data_path.name,
            "sha256": sha256_of(data_path),
            "encoding": read_meta["encoding"],
            "delimiter": read_meta["delimiter"],
            "engine": read_meta.get("engine"),
            "decoding_fallback": read_meta.get("decoding_fallback", False),
            "rows": read_meta["rows"],
            "columns": read_meta["columns"],
            "pandas": pd.__version__,
            "python": sys.version.split()[0],
        },
        "schema": {
            "mapped": mapping,
            "mapping_detail": mapping_detail,
            "ambiguous_or_warned": ambiguous,
            "unmapped": unmapped,
            "coercion_failures": coercion_failures,
            "column_names": read_meta["column_names"],
        },
        "policy": policy,
        "quality": quality,
        "summaries": summaries,
    }

    group_checks = [
        block["reconciles"] for block in summaries.values() if block.get("reconciles") is not None
    ]
    default_amount = policy["default_totals"]["amount"] if policy["default_totals"] else None
    matches_default = [
        abs(block["group_sum_all_amount"] - default_amount) < 0.01
        for block in summaries.values()
        if block.get("group_sum_all_amount") is not None and default_amount is not None
    ]
    results["reconciliation"] = {
        "groups_checked": len(group_checks),
        "all_group_sums_match_overall": all(group_checks) if group_checks else None,
        "groups_compared_to_default_totals": len(matches_default),
        "all_group_sums_match_default_totals": all(matches_default) if matches_default else None,
    }

    json_path = write_json(results, out_dir)
    figures = make_figures(typed_default, mapping, out_dir, policy_label=default)
    results["figures"] = figures
    write_json(results, out_dir)  # 补上 figures 字段
    md_path = write_summary_md(results, out_dir)

    print(f"[OK] 读取 {read_meta['rows']} 行 × {read_meta['columns']} 列 | 编码={read_meta['encoding']} | 分隔符={read_meta['delimiter']!r}")
    print(f"[OK] 语义映射 {json.dumps(mapping, ensure_ascii=False)}")
    if unmapped:
        print(f"[WARN] 未映射列：{unmapped}")
    if ambiguous:
        print(f"[WARN] 映射歧义/口径风险：{json.dumps(ambiguous, ensure_ascii=False)}")
    if coercion_failures:
        print(f"[WARN] 解析失败：{json.dumps({k: v['parse_failed'] for k, v in coercion_failures.items()}, ensure_ascii=False)}")
    print(f"[OK] 默认口径 totals：{json.dumps(policy['default_totals'], ensure_ascii=False)}")
    print(f"[OK] 结果：{json_path}")
    print(f"[OK] 摘要：{md_path}")
    for fig in figures:
        print(f"[OK] 图：{fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
