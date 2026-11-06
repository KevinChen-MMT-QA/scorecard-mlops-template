# core/preprocess.py
"""数据预处理：训练/测试切分、特征清洗与质量筛选、建模数据流水线。

训练与评估共用 prepare_dataset，保证两侧数据口径一致。
"""
import hashlib

import numpy as np
import pandas as pd

from core.config import feature_fields, valid_flag_values
from core.data import load_raw_features, load_sample_ids
from core.rules import veto_config, veto_mask


def assign_split(df, cfg):
    """按主键哈希稳定划分训练/测试集（按 y 分层，保证正负样本比例一致）"""
    pk = cfg["source"]["primary_key"]
    ratio = float(cfg["data"].get("train_ratio", 0.7))
    seed = str(cfg["data"].get("split_seed", 42))

    def _bucket(value):
        digest = hashlib.md5((seed + str(value)).encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 10000

    df["split_tag"] = "train"
    for label in sorted(df["y"].unique()):
        mask = df["y"].eq(label).to_numpy()
        group = df.loc[mask]
        buckets = group[pk].map(_bucket).to_numpy()
        order = np.argsort(buckets, kind="mergesort")
        n_train = int(round(ratio * len(group)))
        tags = np.array(["test"] * len(group), dtype=object)
        tags[order[:n_train]] = "train"
        df.loc[mask, "split_tag"] = tags
    return df


def clean_features(df, cfg):
    """按特征类型清洗，并在训练集上评估质量（缺失率 / 唯一值 / 类别基数）"""
    prep = cfg.get("preprocessing") or {}
    miss_thr = float(prep.get("missing_rate_threshold", 0.8))
    max_card = int((prep.get("category") or {}).get("max_cardinality", 30))
    valid_flag = valid_flag_values(cfg)
    train_mask = df["split_tag"].eq("train").to_numpy()

    rows, status = [], {}
    for field in feature_fields(cfg):
        name, ftype = field["name"], field["feature_type"]
        invalid = list(field.get("invalid_values") or [])
        series = df[name]

        if ftype == "numeric":
            series = pd.to_numeric(series, errors="coerce")
            if invalid:
                series = series.mask(series.isin([float(v) for v in invalid]))
            df[name] = series
        elif ftype == "flag":
            series = series.astype("string").str.strip()
            series = series.mask(~series.isin(valid_flag))
            df[name] = series
        else:
            series = series.astype("string").str.strip()
            series = series.mask(series.eq(""))
            if invalid:
                series = series.mask(series.isin([str(v) for v in invalid]))
            df[name] = series

        train_series = df.loc[train_mask, name]
        miss_rate = float(train_series.isna().mean()) if len(train_series) else 0.0
        n_unique = int(train_series.dropna().nunique())

        if miss_rate > miss_thr:
            st, reason = "缺失率高剔除", f"训练集缺失率 {miss_rate:.2%} > 阈值 {miss_thr:.0%}"
        elif n_unique <= 1:
            st, reason = "常量剔除", f"训练集唯一值数 {n_unique}"
        elif ftype == "category" and n_unique > max_card:
            st, reason = "类别基数超限剔除", f"训练集类别数 {n_unique} > {max_card}"
        else:
            st, reason = "可用", ""
        status[name] = st
        rows.append({
            "字段名": name,
            "中文名": field.get("cn_name", ""),
            "特征类型": ftype,
            "训练集缺失率": round(miss_rate, 6),
            "训练集唯一值数": n_unique,
            "预处理状态": st,
            "说明": reason,
        })
    return df, pd.DataFrame(rows), status


def prepare_dataset(cfg, bizdate=None):
    """建模数据流水线（训练/评估共用同一口径）

    1) 读取人工打标结果（正负样本 csv -> 主键 + y）
    2) 按打标主键取特征（mock 逐行生成 / spark 从 Hive 过滤，spark 模式需传 bizdate）
       同时读取否决/加分规则字段，但不作为入模特征
    3) 按主键哈希 + 分层切分训练集/测试集
    4) 按否决规则剔除训练集命中客户（测试集保留，用于评估 E 等级）
    5) 在训练集上做特征级清洗与质量筛选
    """
    df = assign_split(load_raw_features(cfg, load_sample_ids(cfg), bizdate=bizdate), cfg)

    veto = veto_config(cfg)
    if veto.get("enabled") and veto.get("exclude_from_train"):
        hit = veto_mask(df, cfg)
        is_train = df["split_tag"].eq("train").to_numpy()
        removed = int((hit & is_train).sum())
        kept = int((hit & ~is_train).sum())
        print(f"=== 否决规则：训练集命中 {removed} 条已剔除，测试集命中 {kept} 条保留用于评估 ===")
        df = df[~(hit & is_train)].reset_index(drop=True)

    return clean_features(df, cfg)
