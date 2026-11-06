# core/evaluator.py
"""评分卡评估核心：解析并本地执行评分卡 SQL，计算 AUC / KS / PSI 与 A/B/C/D/E 分层。

本地执行器不依赖 Spark / DuckDB，按 SQL CASE 分支的语义在 pandas 中等价执行；
E 客户（否决命中）不参与区分度指标，单独在分层表中统计。
"""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core.rules import rule_mask, veto_mask

NUM_PATTERN = r"[-+]?\d+(?:\.\d+)?"
LEVEL_ORDER = ["A", "B", "C", "D", "E"]


# ====================== 评分卡 SQL 解析与本地执行 ======================
def parse_scorecard_sql(sql_text):
    """解析由 core/sql_builder.py 生成的评分卡 SQL，提取基础分项、各字段 CASE 分值、ABCD 阈值

    只在 base CTE 内查找分值 CASE（END AS sc_xxx），跳过 engineered CTE 中的特征衍生 CASE。
    """
    # 截断到 base AS (...) 之后，避免 engineered CTE 的 CASE 被正则误匹配
    base_marker = sql_text.find("base AS (")
    if base_marker >= 0:
        sql_text = sql_text[base_marker:]

    base_match = re.search(r"ROUND\((" + NUM_PATTERN + r") \+ sc_", sql_text)
    if not base_match:
        raise ValueError("SQL 中未找到基础分项 ROUND(base + sc_...)")
    base_component = float(base_match.group(1))

    blocks = re.findall(r"CASE\s+(.*?)\s+END AS sc_(\w+)", sql_text, flags=re.S)
    if not blocks:
        raise ValueError("SQL 中未找到字段分值 CASE 块（END AS sc_xxx）")

    parsed_blocks = []
    for body, field in blocks:
        branches = []
        for cond, val in re.findall(r"WHEN (.+?) THEN (" + NUM_PATTERN + r")(?=\s+WHEN|\s+ELSE|\s*$)",
                                    body, flags=re.S):
            branches.append((cond.strip(), float(val)))
        else_match = re.search(r"ELSE (" + NUM_PATTERN + r")\s*$", body.strip())
        if else_match:
            branches.append((None, float(else_match.group(1))))
        if not branches:
            raise ValueError(f"字段 {field} 的 CASE 块未解析到任何分值")
        parsed_blocks.append((field, branches))

    levels = [(float(th), lab) for th, lab in
              re.findall(r"WHEN score >= (" + NUM_PATTERN + r") THEN '([A-D])'", sql_text)]
    if len(levels) != 3:
        raise ValueError(f"SQL 中 ABCD 分层阈值解析失败，期望 3 档，实际 {len(levels)} 档")

    return {"base": base_component, "blocks": parsed_blocks, "levels": levels}


def _parse_string_list(text):
    return [v.replace("''", "'") for v in re.findall(r"'((?:[^']|'')*)'", text)]


def _eval_atom(df, atom):
    """执行单个 SQL 条件原子（仅支持评分卡 SQL 使用的条件形态）"""
    atom = atom.strip()
    m = re.fullmatch(r"(\w+) IS NULL", atom)
    if m:
        return df[m.group(1)].isna().to_numpy(dtype=bool)

    m = re.fullmatch(r"TRIM\((\w+)\) = ''", atom)
    if m:
        series = df[m.group(1)].astype("string").str.strip()
        return series.eq("").fillna(False).to_numpy(dtype=bool)

    m = re.fullmatch(r"TRIM\((\w+)\) IN \((.+)\)", atom, flags=re.S)
    if m:
        series = df[m.group(1)].astype("string").str.strip()
        return series.isin(_parse_string_list(m.group(2))).fillna(False).to_numpy(dtype=bool)

    m = re.fullmatch(r"TRIM\((\w+)\) NOT IN \((.+)\)", atom, flags=re.S)
    if m:
        series = df[m.group(1)].astype("string").str.strip()
        mask = ~series.isin(_parse_string_list(m.group(2)))
        return np.asarray(mask.fillna(False), dtype=bool)

    m = re.fullmatch(r"(\w+) IN \((" + NUM_PATTERN + r"(?:,\s*" + NUM_PATTERN + r")*)\)", atom, flags=re.S)
    if m:
        nums = [float(v) for v in re.findall(NUM_PATTERN, m.group(2))]
        return pd.to_numeric(df[m.group(1)], errors="coerce").isin(nums).to_numpy(dtype=bool)

    m = re.fullmatch(r"(\w+) <= (" + NUM_PATTERN + r")", atom)
    if m:
        series = pd.to_numeric(df[m.group(1)], errors="coerce")
        return (series <= float(m.group(2))).to_numpy(dtype=bool)

    raise ValueError(f"无法解析 SQL 条件：{atom}")


def _eval_condition(df, condition):
    mask = np.zeros(len(df), dtype=bool)
    for atom in re.split(r"\s+OR\s+", condition.strip()):
        mask |= _eval_atom(df, atom)
    return mask


def apply_business_rules(raw_score, df, cfg, thresholds):
    """否决 + 负分归零 + 两级加分 + 分级；E 客户分数照常保留"""
    score = pd.Series(np.asarray(raw_score, dtype=float), index=df.index)
    veto_hit = veto_mask(df, cfg)
    base_pos = score.clip(lower=0.0)

    bonus = (cfg.get("rules") or {}).get("bonus") or {}
    points = pd.Series(0.0, index=df.index)
    percent = pd.Series(0.0, index=df.index)
    max_total = float(bonus.get("max_total", 20))

    if bonus.get("enabled"):
        for rule in bonus.get("score_rules") or []:
            points += rule_mask(df, rule).astype(float) * float(rule.get("points", 0))
        for rule in bonus.get("percent_rules") or []:
            percent += rule_mask(df, rule).astype(float) * float(rule.get("percent", 0))

    stage1 = base_pos + points
    percent_bonus = (stage1 * percent).round(2)
    bonus_total = (points + percent_bonus).clip(upper=max_total)
    final_score = (base_pos + bonus_total).round(2)

    level = pd.Series("D", index=df.index, dtype=object)
    # 从低档到高档赋值，保证高分客户被覆盖为更高等级
    for threshold, label in reversed(thresholds):
        level[(final_score >= threshold).to_numpy()] = label
    level[veto_hit] = "E"
    return final_score, level


def execute_scorecard_sql(sql_text, df, cfg, pk="unt_scl_cdt_cd"):
    """本地执行评分卡 SQL（语义等价于 Hive：CASE WHEN 首个命中生效，分数四舍五入）"""
    parsed = parse_scorecard_sql(sql_text)
    score = pd.Series(float(parsed["base"]), index=df.index, dtype=float)

    for field, branches in parsed["blocks"]:
        part = pd.Series(0.0, index=df.index)
        remain = np.ones(len(df), dtype=bool)
        for condition, value in branches:
            hit = remain if condition is None else (remain & _eval_condition(df, condition))
            part[hit] = value
            remain = remain & ~hit
        score = score + part

    score = score.round(2)
    score, level = apply_business_rules(score, df, cfg, parsed["levels"])

    return pd.DataFrame({
        pk: df[pk].to_numpy(),
        "score": score.to_numpy(),
        "customer_level": level.to_numpy(),
    })


# ====================== 指标计算 ======================
def calc_ks(y_true, prob):
    """KS：按预测违约概率升序累计好/坏分布差的最大值"""
    frame = pd.DataFrame({"y": np.asarray(y_true), "prob": np.asarray(prob)}).sort_values("prob")
    frame["good"] = 1 - frame["y"]
    frame["bad"] = frame["y"]
    frame["cum_good"] = frame["good"].cumsum() / frame["good"].sum()
    frame["cum_bad"] = frame["bad"].cumsum() / frame["bad"].sum()
    return float((frame["cum_good"] - frame["cum_bad"]).abs().max())


def psi_calc(base, comp, bins=10):
    """PSI：以训练集分数分位点为切分边界，比较训练/测试分布"""
    edges = np.unique(np.quantile(np.asarray(base, dtype=float), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    base_bin = pd.cut(pd.Series(base), bins=edges, duplicates="drop")
    comp_bin = pd.cut(pd.Series(comp), bins=edges, duplicates="drop")
    base_pct = base_bin.value_counts(normalize=True, sort=False)
    comp_pct = comp_bin.value_counts(normalize=True, sort=False)
    base_pct, comp_pct = base_pct.align(comp_pct, fill_value=0.0)
    base_pct = base_pct.clip(lower=1e-6)
    comp_pct = comp_pct.clip(lower=1e-6)
    return float(((comp_pct - base_pct) * np.log(comp_pct / base_pct)).sum())


def abcd_table(frame):
    """按 SQL 输出的客户等级统计分数区间、样本量与坏客户率"""
    stat = frame.groupby("customer_level").agg(
        cnt=("y", "count"),
        bad_cnt=("y", "sum"),
        bad_rate=("y", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    )
    stat = stat.reindex(LEVEL_ORDER).dropna(subset=["cnt"])
    stat["score_share"] = stat["cnt"] / stat["cnt"].sum()
    stat = stat.reset_index().rename(columns={
        "customer_level": "客户等级",
        "cnt": "样本数",
        "bad_cnt": "坏客户数",
        "bad_rate": "坏客户率",
        "min_score": "最低分",
        "max_score": "最高分",
        "score_share": "样本占比",
    })
    for col in ("坏客户率", "样本占比"):
        stat[col] = stat[col].round(6)
    return stat
