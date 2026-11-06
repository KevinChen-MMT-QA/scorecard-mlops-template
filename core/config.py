# core/config.py
"""场景配置：YAML 加载、相对路径解析、字段清单校验与全局常量。

设计要点：
1. scenario.yaml 维护场景/流程配置，fields.csv 维护字段清单，源表字段变更时只改这两个文件；
2. 所有相对路径统一按 scenario.yaml 所在目录解析，与启动目录无关；
3. 主键、字段清单、基础参数在此校验，规则校验委托 core.rules（避免模块循环依赖）。
"""
import os

import pandas as pd
import yaml

PRIMARY_KEY = "unt_scl_cdt_cd"
MISSING_LABEL = "MISSING"
FIELD_CSV_COLUMNS = ["name", "cn_name", "data_type", "feature_type"]
FEATURE_TYPES = {"numeric", "flag", "category"}
VALID_RULE_OPS = {">", ">=", "<", "<=", "=", "!="}


def load_config(path):
    """读取场景 YAML 与字段清单，并将相对路径统一按场景目录解析"""
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["scenario"] = cfg.get("scenario") or os.path.basename(base_dir)
    cfg["_scenario_dir"] = base_dir

    fields_path = cfg.get("fields_file")
    if not fields_path:
        raise ValueError(f"{path} 缺少 fields_file 配置（字段清单 csv 路径）")
    fields_path = _resolve_path(fields_path, base_dir)
    cfg["fields"] = _load_fields_csv(fields_path)
    for f in cfg["fields"]:
        f["feature_type"] = _resolve_feature_type(f, fields_path)
    cfg["_field_map"] = {f["name"]: f for f in cfg["fields"]}

    data = cfg.get("data") or {}
    cfg["data"] = data
    for key in ("pos_sample_csv", "neg_sample_csv"):
        if data.get(key):
            data[key] = _resolve_path(data[key], base_dir)

    sql = cfg.get("sql") or {}
    cfg["sql"] = sql
    if sql.get("output_file"):
        sql["output_file"] = _resolve_path(sql["output_file"], base_dir)
    sql["report_dir"] = _resolve_path(sql.get("report_dir", "report"), base_dir)

    # 特征衍生 SQL 文件路径解析（可选）
    fe = cfg.get("feature_engineering") or {}
    cfg["feature_engineering"] = fe
    if fe.get("sql_file"):
        fe["sql_file"] = _resolve_path(fe["sql_file"], base_dir)

    _validate_primary_key(cfg, path)
    # 延迟导入：rules 依赖 config 常量，避免模块级循环导入
    from core.rules import validate_rules
    validate_rules(cfg, path)
    return cfg


def _resolve_path(value, base_dir):
    """相对路径按场景目录解析，绝对路径原样返回"""
    return value if os.path.isabs(value) else os.path.join(base_dir, value)


def _load_fields_csv(path):
    """读取字段清单：name / cn_name / data_type / feature_type，每行即一个入模候选字段"""
    if not os.path.exists(path):
        raise ValueError(f"字段清单文件不存在：{path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in FIELD_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"字段清单 {path} 缺少列 {missing}，需要 {FIELD_CSV_COLUMNS}")
    df = df[FIELD_CSV_COLUMNS].apply(lambda col: col.str.strip())
    if df.empty:
        raise ValueError(f"字段清单 {path} 为空")
    if df["name"].eq("").any():
        raise ValueError(f"字段清单 {path} 存在空字段名")
    dup = sorted(df.loc[df["name"].duplicated(), "name"].unique())
    if dup:
        raise ValueError(f"字段清单 {path} 存在重复字段：{dup}")
    return df.to_dict("records")


def _validate_primary_key(cfg, path):
    """主键校验：主键必须固定为 PRIMARY_KEY，且不能出现在字段清单中"""
    src_pk = str((cfg.get("source") or {}).get("primary_key", "")).strip()
    if src_pk != PRIMARY_KEY:
        raise ValueError(
            f"{path} 主键必须为 {PRIMARY_KEY}（保证样本、特征、SQL 输出同一主键），实际为 {src_pk}"
        )
    if src_pk in cfg["_field_map"]:
        raise ValueError(f"{path} 主键 {src_pk} 不应出现在字段清单 {cfg.get('fields_file')} 中")
    part_col = str((cfg.get("source") or {}).get("partition_field", "")).strip()
    if part_col and part_col in cfg["_field_map"]:
        raise ValueError(f"{path} 分区字段 {part_col} 不应出现在字段清单 {cfg.get('fields_file')} 中")


def _resolve_feature_type(field, path="fields.csv"):
    """校验字段清单中的特征类型：只能是 numeric / flag / category"""
    ftype = str(field.get("feature_type", "")).strip().lower()
    if ftype not in FEATURE_TYPES:
        raise ValueError(
            f"字段清单 {path} 字段 {field.get('name')} 的 feature_type 必须为 "
            f"{sorted(FEATURE_TYPES)}，实际为 {ftype!r}"
        )
    return ftype


def feature_fields(cfg):
    """返回字段清单中的全部入模候选字段"""
    return list(cfg["fields"])


def valid_flag_values(cfg):
    """flag 字段的合法取值（默认 0/1），训练清洗与 SQL 生成共用同一口径"""
    prep = cfg.get("preprocessing") or {}
    return [str(v) for v in (prep.get("flag") or {}).get("valid_values", ["0", "1"])]
