# data.py
"""样本与数据接入：人工打标样本读取、Mock 生成、Spark / Hive 特征加载。

设计要点：
1. 正负样本 csv 为人工打标结果：mock 按打标客户生成特征，生产用 Spark 从 Hive 取特征后按打标主键过滤；
2. 两种模式都以打标结果（主键 + 标签 y）为唯一入口，主键固定 unt_scl_cdt_cd；
3. 规则字段随特征一并读取，但不作为入模候选字段。
"""
import numpy as np
import pandas as pd

from core.config import feature_fields
from core.feature_engineering import feature_sql_text
from core.rules import (
    rule_fields,
    veto_condition_fields,
    veto_conditions,
    veto_config,
)


# ============================ 样本读取 ============================
def load_sample_ids(cfg):
    """读取正负样本（csv / txt），主键列固定 unt_scl_cdt_cd，不做列名映射
    data.sample_format：csv 必须有表头；txt 一行一个主键（无表头，首行主键名自动跳过，
    空白行忽略）。校验：表头缺失或主键为空报错；文件内重复主键自动去重；
    同一主键同时出现在正、负样本中（标签冲突）直接报错。
    """
    data = cfg["data"]
    pk = cfg["source"]["primary_key"]
    frames = []
    fmt = str(data.get("sample_format", "csv")).strip().lower()
    if fmt not in ("csv", "txt"):
        raise ValueError(f"data.sample_format 仅支持 csv / txt，实际为 {fmt!r}")
    for path, label in ((data["pos_sample_csv"], 1), (data["neg_sample_csv"], 0)):
        df = _read_txt_sample(path, pk) if fmt == "txt" else _read_csv_sample(path, pk)
        if df[pk].isna().any():
            raise ValueError(f"样本文件 {path} 存在空主键，共 {int(df[pk].isna().sum())} 行")
        df[pk] = df[pk].astype(str).str.strip()
        if df[pk].eq("").any():
            raise ValueError(f"样本文件 {path} 存在空主键，共 {int(df[pk].eq('').sum())} 行")
        dup_in_file = int(df[pk].duplicated().sum())
        if dup_in_file:
            print(f"样本文件 {path} 内重复主键 {dup_in_file} 条，已去重")
            df = df.drop_duplicates(subset=[pk], keep="first")
        df["y"] = label
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    conflict = sorted(out.loc[out[pk].duplicated(), pk].unique())
    if conflict:
        raise ValueError(f"正负样本标签冲突（同一主键同时出现在正/负样本中）：{conflict[:10]}")
    return out[[pk, "y"]].reset_index(drop=True)


def _read_csv_sample(path, pk):
    """csv 样本：必须有表头，且表头包含主键列"""
    header = list(pd.read_csv(path, nrows=0).columns)
    if pk not in header:
        raise ValueError(
            f"样本文件 {path} 表头与主键配置不一致：需要主键列 {pk}，实际表头为 {header}"
        )
    return pd.read_csv(path, usecols=[pk], dtype=str)


def _read_txt_sample(path, pk):
    """txt 样本：一行一个主键、无表头；首行若为主键名自动跳过，空白行忽略"""
    df = pd.read_csv(path, header=None, names=[pk], dtype=str,
                     encoding="utf-8-sig", keep_default_na=False)
    df[pk] = df[pk].str.strip()
    if not df.empty and df[pk].iloc[0] == pk:
        print(f"样本文件 {path} 首行为主键名，已按表头跳过")
        df = df.iloc[1:]
    return df[df[pk].ne("")].reset_index(drop=True)


# ============================ 特征加载 ============================
def load_raw_features(cfg, df_id, bizdate=None):
    """按 data.load_mode 选择 Mock 或 Spark 加载

    两种模式都以人工打标结果 df_id（主键 + 标签 y）为唯一入口：
    - mock（离线测试）：按打标客户逐行生成 mock 特征；
    - spark（生产）：从 Hive 读特征后，用打标主键过滤，只保留有打标的记录。
    """
    mode = str(cfg["data"].get("load_mode", "mock")).lower()
    pos_cnt = int(df_id["y"].sum())
    print(f"=== 人工打标样本：{len(df_id)} 条（正样本 {pos_cnt}，负样本 {len(df_id) - pos_cnt}） ===")
    if mode == "mock":
        print("=== 数据接入：Mock（按打标样本生成，字段与类型均来自场景 fields.csv） ===")
        return build_mock_data(cfg, df_id)
    if mode == "spark":
        print("=== 数据接入：Spark / Hive（按打标主键过滤） ===")
        return load_features_from_spark(cfg, df_id, bizdate=bizdate)
    raise ValueError(f"不支持的 data.load_mode：{mode}")


def load_features_from_spark(cfg, df_id, bizdate=None, spark=None):
    """生产环境 Hive 特征加载（预留代码块，需安装 pyspark）

    口径：按 part_dt 分区取当日快照（分区日期即快照日期），过滤主键为空的记录，
    再用人工打标主键做内连接，只保留有打标的记录；打标了但 Hive 无特征的记录剔除并告警。
    分区内主键重复时不报错，按样本主键保留首条特征，保证训练集主键唯一。
    本地 Mock 模式不会走到这里，不影响无 Spark 环境的调试。

    如果 scenario.yaml 启用了 feature_engineering，直接执行 feature_engineering.sql
    （里面已经包含衍生字段的计算），否则按原来的"单表 SELECT + 按字段清单拼列"逻辑。
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("data.load_mode=spark 需要先安装 pyspark（pip install pyspark）") from exc

    src = cfg["source"]
    spark_cfg = cfg["data"].get("spark") or {}
    bizdate = bizdate or spark_cfg.get("bizdate")
    if bizdate is None or str(bizdate).strip() in ("", "${bizdate}"):
        raise ValueError("spark 模式必须传入具体分区日期：python run.py train --scenario <name> --bizdate <yyyymmdd>，"
                         "或在 data.spark.bizdate 配置具体日期")
    if spark is None:  # pragma: no cover
        app_name = spark_cfg.get("app_name") or f"scorecard_{cfg.get('scenario', 'feature_load')}"
        spark = (
            SparkSession.builder.appName(app_name)
            .enableHiveSupport()
            .getOrCreate()
        )
    pk = src["primary_key"]

    # 走特征衍生 SQL（包含多表 JOIN + 衍生字段）
    fe_cfg = cfg.get("feature_engineering") or {}
    if fe_cfg.get("enabled") and fe_cfg.get("sql_file"):
        sql = feature_sql_text(cfg, bizdate=bizdate)
        print(f"=== 特征衍生：执行 {fe_cfg['sql_file']}（含衍生字段计算） ===")
    else:
        # 原逻辑：单表 SELECT + 按字段清单拼列
        part_col = src["partition_field"]
        table = f"{src.get('database')}.{src['table_name']}" if src.get("database") else src["table_name"]
        load_cols = [pk] + [f["name"] for f in cfg["fields"]]
        for name in rule_fields(cfg):
            if name not in load_cols:
                load_cols.append(name)
        col_sql = ", ".join(load_cols)
        sql = f"""
            SELECT {col_sql}
            FROM {table}
            WHERE {part_col} = '{bizdate}'
              AND {pk} IS NOT NULL
              AND TRIM({pk}) <> ''
        """
        print(f"=== 特征衍生：未启用，走原逻辑（单表 SELECT） ===")

    df_feature = spark.sql(sql).toPandas()
    df_feature[pk] = df_feature[pk].astype(str).str.strip()
    dup = int(df_feature[pk].duplicated().sum())
    if dup:
        print(f"警告：分区 {bizdate} 内主键重复 {dup} 条，按样本主键保留首条特征")
        df_feature = df_feature.drop_duplicates(subset=[pk], keep="first")
    out = df_id.merge(df_feature, on=pk, how="inner")
    unmatched = len(df_id) - len(out)
    print(f"=== 按打标主键过滤：打标 {len(df_id)} 条，Hive 匹配 {len(out)} 条，"
          f"未匹配剔除 {unmatched} 条 ===")
    if len(out) == 0:
        raise RuntimeError(f"打标样本在分区 {bizdate} 的 Hive 数据中无任何匹配记录，请检查主键与分区")
    if unmatched:
        print(f"警告：{unmatched} 条打标记录在 Hive 快照中无特征，已剔除，不参与训练/评估")
    return out


# ============================ Mock 数据（按 YAML 字段生成） ============================
def build_mock_data(cfg, df_id):
    """按人工打标样本生成 mock 数据（行数、主键、标签均取自 df_id，不额外造客户）

    - 每个打标客户生成一行特征，y 直接沿用打标结果（1 正样本 / 0 负样本）
    - flag：0/1 二值；numeric：按精度生成连续值；category：按类别取值生成
    - signal_* 配置为演示用强弱信号（y=1 与 y=0 分布不同）
    - missing_rates 控制字段缺失率，用于演示缺失率剔除与缺失单独成箱
    """
    mock = cfg.get("mock") or {}
    rng = np.random.RandomState(int(mock.get("random_state", 42)))
    n = len(df_id)
    y = df_id["y"].to_numpy()
    df = df_id.copy()

    flag_sig = {it["field"]: it for it in (mock.get("signal_flags") or [])}
    num_sig = {it["field"]: it for it in (mock.get("signal_numeric") or [])}
    cat_sig = {it["field"]: it for it in (mock.get("signal_category") or [])}
    cat_values = mock.get("category_values") or {}
    default_cat = mock.get("default_category_values") or ["A", "B", "C", "D"]
    miss_rates = mock.get("missing_rates") or {}
    default_miss = float(mock.get("default_missing_rate", 0.08))

    for field in feature_fields(cfg):
        name = field["name"]
        ftype = field["feature_type"]
        if ftype == "numeric":
            values = _mock_numeric(field, num_sig.get(name), rng, y, n)
        elif ftype == "flag":
            values = _mock_flag(flag_sig.get(name), rng, y, n)
        else:
            values = _mock_category(cat_sig.get(name), cat_values.get(name, default_cat), rng, y, n)

        miss_rate = float(miss_rates.get(name, default_miss))
        if miss_rate > 0:
            values = pd.Series(values, dtype="object")
            values[rng.rand(n) < miss_rate] = np.nan
        df[name] = values

    # 否决条件字段不属于入模特征，按 mock_hit_rate 生成命中值 / 安全值，便于本地验证 E 等级
    veto = veto_config(cfg)
    if veto.get("enabled"):
        hit_rate = float(veto.get("mock_hit_rate", 0.02))
        conditions = veto_conditions(cfg)
        for name in veto_condition_fields(cfg):
            field_conds = [c for c in conditions if str(c.get("field", "")).strip() == name]
            safe_value, hit_value = _mock_veto_values(name, field_conds, cat_values, rng)
            df[name] = np.where(rng.rand(n) < hit_rate, hit_value, safe_value)

    # 特征衍生：源字段全部就绪后，用 pandas 计算 9 个衍生特征
    fe_cfg = cfg.get("feature_engineering") or {}
    if fe_cfg.get("enabled"):
        from core.feature_engineering import derive_features
        df = derive_features(df, cfg)
        print(f"=== Mock 特征衍生：已按 feature_engineering 配置计算衍生字段 ===")
    return df


def _mock_veto_values(field, conditions, cat_values, rng):
    """按否决条件推导 mock 的命中值/安全值：= 命中该值；!= 命中安全值以外的取值"""
    cond = conditions[0]
    op = str(cond.get("op", "")).strip()
    text = str(cond.get("value", "")).strip()
    if op == "!=":
        pool = [str(v) for v in (cat_values.get(field) or []) if str(v).strip() != text]
        hit = str(rng.choice(pool)) if pool else ("1" if text != "1" else "0")
        return text, hit
    hit = text
    safe = "0" if text != "0" else "1"
    return safe, hit


def _mock_numeric(field, signal, rng, y, n):
    if signal:
        dist = signal.get("dist", "normal")
        if dist == "normal":
            mu = np.where(y == 1, float(signal["bad_mu"]), float(signal["good_mu"]))
            values = mu + float(signal.get("sigma", 1.0)) * rng.normal(size=n)
        elif dist == "lognormal":
            median = np.where(y == 1, float(signal["bad_median"]), float(signal["good_median"]))
            values = median * np.exp(float(signal.get("sigma_log", 1.0)) * rng.normal(size=n))
        elif dist == "poisson":
            lam = np.where(y == 1, float(signal["bad_lambda"]), float(signal["good_lambda"]))
            values = rng.poisson(lam).astype(float)
        else:
            raise ValueError(f"mock.signal_numeric 不支持的 dist：{dist}")
        if signal.get("min") is not None:
            values = np.maximum(values, float(signal["min"]))
        if signal.get("max") is not None:
            values = np.minimum(values, float(signal["max"]))
        return values

    # 无信号字段：按类型/精度生成温和分布，保证可跑通
    dtype = str(field.get("data_type")).lower()
    if dtype in ("int", "bigint"):
        return rng.poisson(3.0, n).astype(float)
    if field.get("precision") == 6:
        return 0.3 + 0.15 * rng.normal(size=n)
    return 1.0e7 * np.exp(1.2 * rng.normal(size=n))


def _mock_flag(signal, rng, y, n):
    p = np.full(n, 0.3)
    if signal:
        p = np.where(y == 1, float(signal["bad_p"]), float(signal["good_p"]))
    return np.where(rng.rand(n) < p, "1", "0")


def _mock_category(signal, values, rng, y, n):
    if signal:
        vals = np.array(signal["values"], dtype=object)
        bad_cdf = np.cumsum(np.asarray(signal["bad_probs"], dtype=float))
        good_cdf = np.cumsum(np.asarray(signal["good_probs"], dtype=float))
        bad_cdf /= bad_cdf[-1]
        good_cdf /= good_cdf[-1]
        u = rng.rand(n)
        idx_bad = np.clip(np.searchsorted(bad_cdf, u, side="right"), 0, len(vals) - 1)
        idx_good = np.clip(np.searchsorted(good_cdf, u, side="right"), 0, len(vals) - 1)
        return np.where(y == 1, vals[idx_bad], vals[idx_good])

    vals = np.array(list(values), dtype=object)
    probs = np.full(len(vals), 1.0 / len(vals))
    return rng.choice(vals, size=n, p=probs)
