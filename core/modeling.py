# core/modeling.py
"""模型训练核心：WOE 矩阵、VIF 逐步剔除、L1 逻辑回归、分值换算与分位阈值。

输出为 SQL 生成的输入（字段分箱规则 + 分值），不落 pkl、不产出推理代码。
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from core.binning import bin_labels_for_rule, score_with_rules, woe_map_for_rule


def build_woe_frame(df, rules):
    """按规则生成 WOE 矩阵（列名 {字段}_woe）"""
    out = pd.DataFrame(index=df.index)
    for rule in rules:
        labels = bin_labels_for_rule(rule, df[rule["field"]])
        out[f"{rule['field']}_woe"] = labels.map(woe_map_for_rule(rule)).astype(float)
    return out


def calc_vif_frame(woe_df, columns, threshold):
    """逐步计算 VIF 并剔除共线性最高变量，返回保留列与 VIF 记录"""
    keep = list(columns)
    vif_values = {}
    if len(keep) <= 1:
        if keep:
            vif_values[keep[0]] = 1.0
        return keep, vif_values

    while len(keep) > 1:
        x = woe_df[[f"{c}_woe" for c in keep]].astype(float)
        const_cols = [c for c in keep if x[f"{c}_woe"].nunique() <= 1]
        if const_cols:
            for col in const_cols:
                keep.remove(col)
                vif_values[col] = float("inf")
            if len(keep) <= 1:
                if keep:
                    vif_values[keep[0]] = 1.0
                break
            continue
        values = [float(variance_inflation_factor(x.values, i)) for i in range(x.shape[1])]
        for col, val in zip(keep, values):
            vif_values[col] = val
        if max(values) <= threshold:
            break
        keep.remove(keep[int(np.argmax(values))])
    if len(keep) == 1 and keep[0] not in vif_values:
        vif_values[keep[0]] = 1.0
    return keep, vif_values


def fit_logistic_model(woe_train, columns, y_train, cfg):
    """L1 逻辑回归并换算基准分项：返回系数、有效字段、factor、base_component"""
    model_cfg = cfg["model"]
    x_train = woe_train[[f"{c}_woe" for c in columns]]
    lr = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=float(model_cfg.get("l1_C", 0.1)),
        max_iter=int(model_cfg.get("max_iter", 1000)),
        random_state=int(model_cfg.get("random_state", 42)),
    )
    lr.fit(x_train, y_train)
    coef = dict(zip(columns, lr.coef_[0]))
    final_fields = [c for c in columns if abs(coef[c]) > 0]
    if not final_fields:
        raise RuntimeError("L1 正则将全部系数压缩为 0，请调大 model.l1_C")

    factor = float(model_cfg.get("pdo", 10)) / np.log(2)
    base_component = float(model_cfg.get("base_score", 100)) - factor * float(lr.intercept_[0])
    return coef, final_fields, factor, base_component


def assign_points(final_rules, coef, factor):
    """分值换算：score = base - factor × (β0 + Σβi·WOEi)"""
    for rule in final_rules:
        coefficient = coef[rule["field"]]
        for bin_cfg in rule["bins"]:
            bin_cfg["points"] = round(-factor * coefficient * bin_cfg["woe"], 6)
        rule["missing"]["points"] = round(-factor * coefficient * rule["missing"]["woe"], 6)


def compute_thresholds(train_df, final_rules, base_component):
    """训练集基础分分位点作为 A/B/C/D 阈值，直接写入 SQL（加分后再分级）

    分位口径：A ≥ 90分位（前 10%）、B ≥ 80分位（10%-20%）、C ≥ 60分位（20%-40%）、D < 60分位（剩余 60%）。
    """
    train_score = score_with_rules(train_df, final_rules, base_component)
    q90, q80, q60 = [round(float(v), 2) for v in np.percentile(train_score, [90, 80, 60])]
    t_a = max(q90, round(q80 + 0.01, 2))
    t_b = max(q80, round(q60 + 0.01, 2))
    t_c = q60
    return (t_a, t_b, t_c)
