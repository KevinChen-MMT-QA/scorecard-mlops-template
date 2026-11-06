# 场景使用说明

1. 在 `samples/` 放入人工打标样本：
   - `pos_sample_ids.csv`：正样本，表头 `unt_scl_cdt_cd`
   - `neg_sample_ids.csv`：负样本，表头 `unt_scl_cdt_cd`
2. 修改 `fields.csv`：字段名、中文名、数据类型、特征类型（numeric/flag/category）。
3. 修改 `scenario.yaml`：源表、分区、样本路径、预处理、模型与规则。
4. 执行：
   ```bash
   python run.py train --scenario <name>
   python run.py evaluate --scenario <name>
   ```
5. 生成物统一写入本场景的 `report/` 目录，不会影响其他场景。
