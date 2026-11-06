# core/feature_engineering.py
"""特征衍生模块：pandas 端执行器 + SQL 文件桥接。

训练端（Spark / Mock）共用同一个 derive_features(df, cfg) 入口：
- Spark 模式：直接执行 feature_engineering.sql（core/data.py 处理），本模块提供源字段清单；
- Mock 模式：build_mock_data 生成源字段后，本模块用 pandas 计算 9 个衍生列，
  保证与 Spark/SQL 产物口径完全一致（每份规则在 SQL 文件和本模块各有一份实现）。

SQL 产物端（core/sql_builder.py）把 feature_engineering.sql 内嵌到评分卡 SQL 的 CTE 链中。
"""
import os

import numpy as np
import pandas as pd


# ============================ 衍生入口 ============================
def derive_features(df, cfg):
    """按 scenario.yaml 中的 feature_engineering 配置，对 df 计算衍生特征并原地覆盖。

    调用时机：build_mock_data 生成完所有源字段后，prepare_dataset 清洗之前。
    Spark 模式不走这个函数（直接 spark.sql(feature_sql)），这里只服务 mock 离线训练。
    """
    fe_cfg = cfg.get("feature_engineering") or {}
    if not fe_cfg.get("enabled"):
        return df

    df = _derive_wht_tech_co(df)
    df = _derive_rt_rslt_idx(df)
    df = _derive_rto_dept_ln(df)
    df = _derive_rto_ln_dept(df)
    df = _derive_credit_bal_pct_idx(df)
    df = _derive_fin_idx(df)
    df = _derive_wht_ipo_progress(df)
    df = _derive_wht_rev_growth_rt(df)
    df = _derive_wht_credit_bal_pct(df)
    return df


# ============================ 各衍生规则实现 ============================

def _str_mask(df, field):
    """字符串字段 → 去除首尾空格的 Series（空串视为缺失）"""
    return df[field].astype("string").str.strip()


def _num(df, field):
    """数值字段 → double，非法值视为 NaN"""
    return pd.to_numeric(df[field], errors="coerce")


def _derive_wht_tech_co(df):
    """any_flag：7 个科技标识任一='1' → '1'"""
    sources = [
        "nation_innov_demo_ent_ind", "nation_new_tech_ent_ind",
        "manuf_champ_ent_ind", "prov_srdi_mid_sml_ent_ind",
        "st_mid_sml_ent_ind", "innov_mid_sml_ent_ind",
        "wht_hard_tech",
    ]
    mask = pd.Series(False, index=df.index)
    for src in sources:
        mask |= (_str_mask(df, src) == "1").fillna(False)
    df["wht_tech_co"] = np.where(mask, "1", "0")
    return df


def _derive_rt_rslt_idx(df):
    """map_values：rt_rslt_id → rt_rslt_idx
    TA1/TA2/TA3 → '2'；TB1/TB2 → '1'；其他 → '0'
    """
    s = _str_mask(df, "rt_rslt_id")
    result = pd.Series("0", index=df.index, dtype="string")
    result[s.isin(["TA1", "TA2", "TA3"])] = "2"
    result[s.isin(["TB1", "TB2"])] = "1"
    df["rt_rslt_idx"] = result
    return df


def _derive_rto_dept_ln(df):
    """nested_if：存款日均≥1000万 且 贷款余额>0 且 存款日均≥3倍贷款余额 → '1'"""
    dept = _num(df, "dept_bal_avg_y")
    ln = _num(df, "undisc_ln_bal")
    mask = (dept >= 10000000) & (ln > 0) & (dept >= ln * 3)
    df["rto_dept_ln"] = np.where(mask.fillna(False), "1", "0")
    return df


def _derive_rto_ln_dept(df):
    """nested_if：贷款余额≥1000万 且 贷款余额>100倍存款日均 → '1'"""
    ln = _num(df, "undisc_ln_bal")
    dept = _num(df, "dept_bal_avg_y")
    mask = (ln >= 10000000) & (ln > dept * 100)
    df["rto_ln_dept"] = np.where(mask.fillna(False), "1", "0")
    return df


def _derive_credit_bal_pct_idx(df):
    """nested_if：用信余额占比>0 且 <0.1 → '1'"""
    pct = _num(df, "credit_bal_pct")
    mask = (pct > 0) & (pct < 0.1)
    df["credit_bal_pct_idx"] = np.where(mask.fillna(False), "1", "0")
    return df


def _derive_fin_idx(df):
    """nested_if：财务指标评分（百分制）
    资产负债率<65% 且 净利润>0 且 营收≥2500万 且 利润率≥10% → 100
    资产负债率<75% 且 净利润>0 且 营收≥1500万 且 利润率≥8%  → 80
    资产负债率<80% 且 净利润>0 且 营收≥500万              → 60
    否则                                                    → 40
    """
    lia = _num(df, "ast_lia_rt")
    pft = _num(df, "nt_pft")
    rev = _num(df, "biz_tot_incm")
    pm = _num(df, "sl_pm_rt")

    result = np.full(len(df), 40.0)
    cond3 = (lia < 0.80) & (pft > 0) & (rev >= 5000000)
    cond2 = (lia < 0.75) & (pft > 0) & (rev >= 15000000) & (pm >= 0.08)
    cond1 = (lia < 0.65) & (pft > 0) & (rev >= 25000000) & (pm >= 0.10)
    result[cond3.fillna(False)] = 60
    result[cond2.fillna(False)] = 80
    result[cond1.fillna(False)] = 100
    df["fin_idx"] = result
    return df


def _derive_wht_ipo_progress(df):
    """any_value_nonempty：ipo_progress 非空 → '1'"""
    s = _str_mask(df, "ipo_progress")
    mask = s.notna() & s.ne("")
    df["wht_ipo_progress"] = np.where(mask, "1", "0")
    return df


def _derive_wht_rev_growth_rt(df):
    """nested_if：rev_growth_rt > 0.1 → '1'"""
    rt = _num(df, "rev_growth_rt")
    df["wht_rev_growth_rt"] = np.where((rt > 0.1).fillna(False), "1", "0")
    return df


def _derive_wht_credit_bal_pct(df):
    """nested_if：credit_bal_pct > 0 且 <= 0.1 → '1'"""
    pct = _num(df, "credit_bal_pct")
    mask = (pct > 0) & (pct <= 0.1)
    df["wht_credit_bal_pct"] = np.where(mask.fillna(False), "1", "0")
    return df


# ============================ 元数据 ============================

# 每个衍生特征依赖的源字段清单（用于 Spark 多读、Mock 确保字段存在）
DERIVATION_SOURCES = {
    "wht_tech_co": [
        "nation_innov_demo_ent_ind", "nation_new_tech_ent_ind",
        "manuf_champ_ent_ind", "prov_srdi_mid_sml_ent_ind",
        "st_mid_sml_ent_ind", "innov_mid_sml_ent_ind",
        "wht_hard_tech",
    ],
    "rt_rslt_idx": ["rt_rslt_id"],
    "rto_dept_ln": ["dept_bal_avg_y", "undisc_ln_bal"],
    "rto_ln_dept": ["undisc_ln_bal", "dept_bal_avg_y"],
    "credit_bal_pct_idx": ["credit_bal_pct"],
    "fin_idx": ["ast_lia_rt", "nt_pft", "biz_tot_incm", "sl_pm_rt"],
    "wht_ipo_progress": ["ipo_progress"],
    "wht_rev_growth_rt": ["rev_growth_rt"],
    "wht_credit_bal_pct": ["credit_bal_pct"],
}


def derivation_source_fields():
    """所有衍生特征依赖的源字段（去重）。供 core/data.py 多 SELECT。"""
    seen = set()
    result = []
    for fields in DERIVATION_SOURCES.values():
        for f in fields:
            if f not in seen:
                seen.add(f)
                result.append(f)
    return result


def feature_sql_text(cfg, bizdate=None):
    """读取 feature_engineering.sql 文件内容并替换 ${bizdate} 占位符。

    Spark 模式和 SQL 产物生成都用这个函数拿到 SQL 原文。
    - Spark：直接 spark.sql(sql_text) 执行后 toPandas()
    - SQL 产物：作为 engineered CTE 内嵌到评分卡 SQL 中
    """
    fe_cfg = cfg.get("feature_engineering") or {}
    sql_path = fe_cfg.get("sql_file")
    if not sql_path:
        raise ValueError("feature_engineering.enabled=true 但缺少 sql_file 配置")

    base_dir = cfg.get("_scenario_dir", os.path.dirname(os.path.abspath("scenario.yaml")))
    abs_path = sql_path if os.path.isabs(sql_path) else os.path.join(base_dir, sql_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"feature_engineering.sql_file 不存在：{abs_path}")

    with open(abs_path, encoding="utf-8") as f:
        text = f.read()

    if bizdate is not None:
        text = text.replace("${bizdate}", str(bizdate))
    return text
