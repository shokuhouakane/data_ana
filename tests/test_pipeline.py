"""管线回归测试：用临时生成的合成 CSV 验证读取层与口径层。

这些文件均为测试时生成的合成数据，不是任务数据。
运行：venv/Scripts/python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "run_all.py"


def run_pipeline(csv_path: Path, out_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    # Windows 子进程默认 cp936；显式指定 UTF-8，测试才能在中文输出下稳定断言
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, str(RUN), "--data", str(csv_path), "--out", str(out_dir), *extra],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_gbk_and_renamed_columns(self):
        csv = self.dir / "gbk.csv"
        text = (
            "编号,日期,大区,品类,商品,渠道,售价,件数,营收,总成本,评分,退货\n"
            "A001,2026/01/05,华东,数码,耳机,线上,299,2,598,180,4.5,否\n"
            "A002,2026/02/06,华北,家居,台灯,线下,159,1,159,90,4.0,是\n"
        )
        csv.write_bytes(text.encode("gb18030"))
        out = self.dir / "out"
        proc = run_pipeline(csv, out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads((out / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(result["meta"]["encoding"], "gb18030")
        self.assertEqual(result["schema"]["mapped"].get("amount"), "营收")
        self.assertEqual(result["schema"]["mapped"].get("order_id"), "编号")
        self.assertEqual(result["quality"]["row_count"], 2)

    def test_semicolon_english_headers(self):
        csv = self.dir / "semi.csv"
        csv.write_text(
            "order_id;date;region;category;channel;price;quantity;revenue;cost;returned\n"
            'B1;2026-03-01;East;Digital;Online;1,299.00;2;2,598.00;1,000;No\n',
            encoding="utf-8",
        )
        out = self.dir / "out"
        proc = run_pipeline(csv, out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads((out / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(result["meta"]["delimiter"], ";")
        self.assertEqual(result["schema"]["mapped"].get("amount"), "revenue")
        self.assertEqual(result["policy"]["default_totals"]["amount"], 2598.0)
        self.assertEqual(result["quality"]["amount_consistency"]["mismatch_rows"], 0)

    def test_unit_cost_header_is_flagged(self):
        csv = self.dir / "unitcost.csv"
        csv.write_text(
            "订单编号,订单日期,单价,数量,销售额,Unit Cost\n"
            "C1,2026-04-01,100,2,200,60\n",
            encoding="utf-8",
        )
        out = self.dir / "out"
        proc = run_pipeline(csv, out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads((out / "results.json").read_text(encoding="utf-8"))
        detail = result["schema"]["mapping_detail"]["cost"]
        self.assertEqual(detail["column"], "Unit Cost")
        self.assertTrue(detail["warning"])
        self.assertTrue(result["quality"]["cost_semantics"]["likely_unit_cost"])

    def test_unparsable_numbers_are_counted(self):
        csv = self.dir / "badnum.csv"
        csv.write_text(
            "订单编号,订单日期,单价,数量,销售额\n"
            "D1,2026-05-01,-,2,200\n",
            encoding="utf-8",
        )
        out = self.dir / "out"
        proc = run_pipeline(csv, out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads((out / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(result["schema"]["coercion_failures"]["price"]["parse_failed"], 1)
        self.assertEqual(result["quality"]["amount_consistency"]["non_comparable_rows"], 1)

    def test_summaries_use_default_policy(self):
        csv = ROOT / "tests" / "fixtures" / "synthetic_orders.csv"
        out = self.dir / "out"
        proc = run_pipeline(csv, out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads((out / "results.json").read_text(encoding="utf-8"))
        default_amount = result["policy"]["default_totals"]["amount"]
        self.assertEqual(default_amount, 16334.0)
        for role, block in result["summaries"].items():
            if "group_sum_all_amount" in block:
                self.assertAlmostEqual(block["group_sum_all_amount"], default_amount, places=2, msg=role)
                self.assertTrue(block["reconciles"], role)
        self.assertTrue(result["reconciliation"]["all_group_sums_match_default_totals"])
        monthly = {row["month"]: row for row in result["summaries"]["monthly"]["rows"]}
        self.assertFalse(monthly["2026-08"]["complete_month"])
        self.assertLess(monthly["2026-08"]["coverage_days"], monthly["2026-08"]["days_in_month"])

    def test_missing_required_role_exits_nonzero(self):
        csv = self.dir / "norole.csv"
        csv.write_text("订单编号,订单日期\nE1,2026-06-01\n", encoding="utf-8")
        proc = run_pipeline(csv, self.dir / "out")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("缺少必需语义角色", proc.stderr)


if __name__ == "__main__":
    unittest.main()
