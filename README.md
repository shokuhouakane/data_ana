# data_ana — 限时数据分析任务

> 状态：等待任务数据发放。分析骨架已就绪，拿到数据后按下面的命令直接跑，再填写结论。

## 我的判断

（待数据发放后填写：从数据里看到了什么、根问题是什么、依据哪几条证据。）

## 做了什么

（待填写：口径定义、清洗规则、分析范围、产出物。）

## 怎么证明有用

（待填写：每个结论对应的数字、图表与复现命令。）

## 复现方式

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 正式数据
python run_all.py --data data/<任务数据>.csv --out results
```

输出：

- `results/results.json`：机器可读结果。README 中的数字只引用这里，不手抄终端输出。
- `results/summary.md`：数据体检摘要（重复、缺失、金额不一致、退货、成本语义、日期）。
- `results/figures/`：概览图；正式结论图会在分析层另写。

`results.json` 里包含：

- `meta`：数据文件 sha256、编码、分隔符、行数、运行环境版本；
- `schema`：列名到语义角色的映射、歧义/风险标记、解析失败计数、未映射列；
- `policy`：去重（3）× 退货（计入/剔除/净额）× 金额（2）共 18 种组合及各自 totals，默认口径另列；每种组合都带 `excluded_amount`、`returned_amount_included`；
- `quality`：每类异常的**影响金额**与可回查的订单号；
- `summaries`：按**默认口径过滤后**的数据生成，与 `default_totals` 同源；每块带 `policy`/`truncated`/`reconciles`；
- `summaries.monthly`：每月带 `complete_month` 与 `coverage_days`，月度图用 `*` 标出不完整月份。

管线回归测试：

```bash
python -m unittest discover -s tests -v
```

## 数据说明

- `tests/fixtures/synthetic_orders.csv`：**合成测试数据**，仅用于验证读取/体检/绘图/导出链路，不参与任何结论，不作为结论依据。
- 正式任务数据放在 `data/`，原样提交，不做手工修改。

## 目录

```
analysis/          读取、列映射、体检、口径、导出
run_all.py         一键入口
tests/             管线回归测试（合成数据）
tests/fixtures/    合成测试夹具
```
