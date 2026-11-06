# 使用说明

评分卡模板以场景为单位组织：引擎代码与场景无关，样本、字段、配置、产物按场景隔离。

## 1. 命令

```bash
python run.py list                                # 列出场景
python run.py init --scenario auto_loan           # 从 _template 新建场景
python run.py train    --scenario industry_finance
python run.py evaluate --scenario industry_finance
python run.py pipeline --scenario industry_finance   # 训练 + 评估
```

生产 Spark 模式需传分区日期：

```bash
python run.py train    --scenario industry_finance --bizdate 20261031
python run.py evaluate --scenario industry_finance --bizdate 20261031
```

也可直接指定配置（优先于 `--scenario`）：

```bash
python run.py pipeline --config scenarios/industry_finance/scenario.yaml --bizdate 20261031
```

## 2. 场景目录

```text
scenarios/<name>/
├── scenario.yaml   # 源表、样本、预处理、模型、SQL、规则、mock
├── fields.csv      # 入模候选字段清单
├── samples/        # 人工打标正负样本
├── report/         # 运行产物
└── columns.txt     # 源表字段参考（可选）
```

所有相对路径都以 `scenario.yaml` 所在目录为基准，与启动目录无关。

## 3. 新增场景

1. `python run.py init --scenario <name>` 复制模板；
2. 修改 `scenario.yaml` 的库表、分区、样本路径与模型参数；
3. 修改 `fields.csv` 的候选字段；
4. 在 `samples/` 放入正负样本；
5. 按需配置 `rules`（否决 / 加分）；
6. 先 `train` 验证，再 `evaluate` 出报告。

## 4. 关键配置

| 配置块 | 说明 |
| --- | --- |
| `source` | Hive 库表、固定主键 `unt_scl_cdt_cd`、分区字段 |
| `fields_file` | 字段清单 csv 路径 |
| `data` | `load_mode: mock/spark`、样本路径与格式、训练测试比例 |
| `preprocessing` | 缺失率阈值、分箱参数、类别基数上限、flag 合法值 |
| `model` | IV/VIF 阈值、L1 强度、基准分、PDO、分数小数位 |
| `sql` | SQL 输出路径、报告目录、分数列名、等级列名 |
| `rules` | 否决 `veto.any`、加分 `bonus.score_rules/percent_rules` |
| `mock` | 仅 mock 模式的演示数据配置，生产可删除 |

主键固定的原因：样本、特征、SQL 输出必须使用同一主键，才能保证训练集与线上出分一一对应。
样本格式由 `data.sample_format` 控制：`csv` 必须有表头；`txt` 一行一个主键，首行主键名自动跳过。

## 5. 运行产物

均写入场景 `report/`：

| 文件 | 内容 |
| --- | --- |
| `scorecard.sql` | Hive 单条 SELECT，分箱分值内联，可直接执行 |
| `score_detail.csv` | 分箱、样本量、坏率、WOE、系数、分值贡献、字段状态 |
| `feature_quality.csv` | 缺失率、唯一值数、IV、VIF、筛选状态 |
| `eval_result.csv` | AUC / KS / PSI、样本量、E 户数、入模字段数 |
| `abcd_rule.csv` | A/B/C/D/E 分数区间、样本量、坏客户率、占比 |
| `model_report.md` | 模型交付说明 |

线上评分：

```bash
hive -hivevar bizdate=20261101 -f scenarios/<name>/report/scorecard.sql
```

## 6. 扩展点

- 新增数据源：修改 `core/data.py` 的 `load_raw_features` 分支；
- 新增分箱策略：修改 `core/binning.py` 的 `fit_bin_rule`；
- 新增模型：在 `core/modeling.py` 中替换训练与分值换算，保持输出字段规则结构；
- 新增报告：在 `core/reporting.py` 中增加渲染函数，`train/evaluate` 只做编排。
