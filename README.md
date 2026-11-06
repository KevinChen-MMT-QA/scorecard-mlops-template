# 评分卡模型模板

> 基于逻辑回归的评分卡模板：卡方分箱 / WOE / IV / VIF / L1，训练后直接产出**可在 Hive 执行的评分卡 SQL**，
> 不保存 pkl，也不提供推理代码。业务规则（一票否决、加分）内联进 SQL。
>
> 当前内置示例场景：`industry_finance`（产业金融营销优先级）。

## 项目结构

```text
.
├── run.py                          # 统一入口：list / init / train / evaluate / pipeline
├── core/                           # 与场景无关的评分卡引擎
│   ├── config.py                   # 配置加载、路径解析、字段清单与主键校验
│   ├── data.py                     # 样本读取、Mock 生成、Spark / Hive 特征加载
│   ├── feature_engineering.py      # 特征衍生：pandas 端执行器 + SQL 文件桥接
│   ├── preprocess.py               # 训练/测试切分、特征清洗、建模数据流水线
│   ├── binning.py                  # 卡方分箱、WOE / IV、本地评分
│   ├── rules.py                    # 否决 / 加分规则校验与 pandas 语义
│   ├── modeling.py                 # WOE 矩阵、VIF、L1 逻辑回归、分值换算、分位阈值
│   ├── sql_builder.py              # 评分卡 SQL 生成（内嵌特征衍生 CTE）
│   ├── evaluator.py                # SQL 解析与本地执行、AUC/KS/PSI、分层
│   ├── reporting.py                # 明细 CSV 组装与 Markdown 报告
│   ├── train.py                    # 训练编排
│   └── evaluate.py                 # 评估编排
├── docs/
│   ├── usage.md                    # 使用说明：命令、场景、配置、产物、扩展点
│   └── methodology.md              # 建模方法：分箱/筛选/换算/规则/评估口径
├── tests/                          # 配置路径、规则语义、CLI 冒烟测试
├── scenarios/
│   ├── _template/                  # 新场景脚手架（init 复制用）
│   └── industry_finance/           # 示例场景，自包含
│       ├── scenario.yaml           # 源表、样本、预处理、模型、SQL、规则、Mock、特征衍生
│       ├── feature_engineering.sql  # 特征衍生 SQL（训练端和 SQL 产物端共用）
│       ├── fields.csv              # 字段清单（含衍生字段）
│       ├── columns.txt             # 源表字段参考
│       ├── samples/                # 人工打标正负样本
│       ├── report/                 # 运行产物（SQL / CSV / Markdown）
│       └── ACCEPTANCE.md           # 业务验收与技术交底材料
└── README.md
```

## 快速开始

```bash
# 查看已注册场景
python run.py list

# 训练 + 评估示例场景
python run.py pipeline --scenario industry_finance

# 分步执行
python run.py train    --scenario industry_finance
python run.py evaluate --scenario industry_finance
```

生产 Spark 模式需传分区日期：

```bash
python run.py train    --scenario industry_finance --bizdate 20261031
python run.py evaluate --scenario industry_finance --bizdate 20261031
```

也可以直接指定配置：

```bash
python run.py pipeline --config scenarios/industry_finance/scenario.yaml --bizdate 20261031
```

生产评分（SQL 即模型）：

```bash
hive -hivevar bizdate=20261101 -f scenarios/industry_finance/report/scorecard.sql
```

依赖：`pandas`、`numpy`、`scikit-learn`、`statsmodels`、`PyYAML`；Spark 模式另需 `pyspark`。

## 新建场景

```bash
python run.py init --scenario auto_loan
```

该命令会复制 `scenarios/_template/` 到 `scenarios/auto_loan/`，并替换场景名。随后需要：

1. 修改 `scenario.yaml` 的库表、分区、样本路径与模型参数；
2. 修改 `fields.csv` 的候选字段；
3. 在 `samples/` 放入人工打标正负样本；
4. 按需配置 `rules`（否决 / 加分）。

## 场景配置约定

每个场景目录自包含，**所有相对路径都以 `scenario.yaml` 所在目录为基准**，与启动目录无关。

| 配置块 | 作用 |
| --- | --- |
| `scenario` | 场景标识，与目录名一致 |
| `source` | Hive 库表、固定主键 `unt_scl_cdt_cd`、分区字段 |
| `fields_file` | 字段清单 csv，默认 `fields.csv` |
| `feature_engineering` | 特征衍生：`enabled=true` + `sql_file` 指定 Hive SQL 文件；训练端（Spark）和 SQL 产物端共用同一份 SQL；mock 模式由 `core/feature_engineering.py` 用 pandas 等价执行 |
| `data` | `load_mode: mock/spark`、样本路径与格式、训练测试比例、切分种子 |
| `preprocessing` | 缺失率阈值、数值分箱参数、类别基数上限、flag 合法取值 |
| `model` | IV/VIF 阈值、L1 强度、基准分、PDO、分数小数位 |
| `sql` | SQL 输出路径、报告目录、分数列名、等级列名 |
| `rules` | 否决（`veto.any`）与加分（`bonus.score_rules/percent_rules`） |
| `mock` | 仅 mock 模式的演示数据配置，生产可删除 |

主键约定：`source.primary_key` 固定为 `unt_scl_cdt_cd`，不得出现在 `fields.csv`；样本表头、
特征表、SQL 输出统一使用该主键，不做别名映射。

样本格式由 `data.sample_format` 控制：

- `csv`（默认）：必须有表头且包含 `unt_scl_cdt_cd`；
- `txt`：一行一个主键、无表头，首行若为主键名会自动跳过。

文件内重复主键自动去重；同一主键同时出现在正负样本中会直接报错。

## 建模流程

```text
人工打标样本（pos/neg）
  → 按主键取特征（mock 逐行生成 / spark 执行 feature_engineering.sql 从 Hive 取数）
  → 特征衍生（feature_engineering.sql 多表 JOIN + 衍生字段；mock 用 pandas 等价执行）
  → 按主键哈希 + 标签分层切分训练/测试集（70/30）
  → 训练集剔除否决命中客户（测试集保留用于评估）
  → 特征清洗：缺失率、常量、类别基数
  → 卡方分箱 → WOE/IV → VIF 逐步剔除 → L1 逻辑回归
  → 分值换算：score = base - PDO/ln2 × (β0 + Σβi·WOEi)
  → 生成评分卡 SQL（内嵌特征衍生 CTE + 分箱分值 + 否决 + 负分归零 + 加分 + 分级）
  → 本地执行 SQL 评估，生成报告
```

业务规则：

- **否决**：`rules.veto.any` 中任一 `{field, op, value}` 命中 → 等级 E，分数照常输出；
  字符串条件仅支持 `=` / `!=`，`!=` 条件缺失也命中；
- **加分**：基础分负分归零 → 分值类加分 → 百分比类加分 → 总加分封顶 `max_total`；
  A/B/C/D 阈值仍按基础分确定，加分后再分级。

## 产物（均写入场景 `report/`）

| 文件 | 内容 |
| --- | --- |
| `scorecard.sql` | Hive 单条 SELECT，分箱分值内联，可直接执行 |
| `score_detail.csv` | 候选字段的分箱、样本量、坏率、WOE、系数、分值贡献、状态 |
| `feature_quality.csv` | 缺失率、唯一值数、IV、VIF、预处理与最终筛选状态 |
| `eval_result.csv` | AUC / KS / PSI、样本量、E 户数、入模字段数 |
| `abcd_rule.csv` | A/B/C/D/E 分数区间、样本量、坏客户率、占比 |
| `model_report.md` | 模型交付说明 |

问答式验收材料见 `scenarios/industry_finance/ACCEPTANCE.md`。
使用说明见 [docs/usage.md](docs/usage.md)，建模口径见 [docs/methodology.md](docs/methodology.md)。

## 测试

```bash
python -m unittest discover -s tests -v
```

覆盖配置相对路径解析、主键与字段校验、否决/加分规则语义、`run.py list/init` 冒烟。

## 评估口径

`core/evaluate.py` 读取场景 `report/scorecard.sql`，由 `core/evaluator.py` 在本地 pandas 中按 CASE 分支等价执行
（不依赖 Spark / DuckDB），然后计算：

- AUC / KS：分数越高越好，用 `-score` 作为违约概率单调替身；
- PSI：训练集 vs 测试集分数分布；
- 分层：按 SQL 输出的 `customer_level` 汇总坏客户率；
- E 客户不参与 AUC/KS/PSI，单独在 ABCD 表中统计。

## 已知限制

- 本地 SQL 评估器只支持本模板生成的 SQL 条件形态（`IS NULL` / `TRIM` / `IN` / `<=`），手改 SQL 后需确认可解析；
- 当前示例场景指标基于小样本 Mock，仅用于流程与规则验证；
- 分箱未引入单调性约束，如需业务可解释性可后续增加单调分箱。
