# 项目说明（给 AI 的常驻上下文）

这是一个 30 分钟限时数据分析任务的应战仓库。**动手前先读**
`docs/superpowers/specs/2026-10-21-data-ana-task-design.md`，里面有完整的需求对齐、分工、时间预算与交付契约。

## 不可违背的约束

- 只能用任务方发放的数据；合成数据必须标明，且不能作为结论依据。
- README 中每个数字都必须由仓库内脚本从原始 CSV 算出，并可从 `results/results.json` 追溯；禁止手抄终端输出。
- 不在数据到达前编造业务结论。
- 任何去重、退货、金额口径必须先声明再计算；体检层只报事实，不静默剔除。
- 推送前让用户确认；提交仓库地址由用户完成。

## 常用命令

```bash
python run_all.py --data data/<csv> --out results
python -m unittest discover -s tests -v
```

解释器用仓库内 `venv/Scripts/python.exe`（Windows）。系统 Python 缺 openpyxl。

## 仓库状态

- 远程：`https://github.com/shokuhouakane/data_ana`（Public，main）
- 当前阶段：骨架与回归测试已就绪，等待任务数据。
- 审查：每批代码改动交给常驻审查子 agent 复核，结论写进 `AI_LOG.md`。
