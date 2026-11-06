# core/rules.py
"""业务规则：否决（一票否决 → E）与加分规则的配置校验、字段收集与 pandas 语义。

规则字段只参与规则判断，不进入入模候选字段；训练、SQL 生成、评估共用本模块口径：
- 否决：任一条件命中即命中；字符串条件仅支持 = / !=，!= 在缺失时也命中；
- 加分：all 条件全部满足、any 条件满足其一；缺失不命中。
"""
import numpy as np
import pandas as pd

from core.config import VALID_RULE_OPS


# ============================ 配置校验 ============================
def validate_rules(cfg, path):
    """校验否决 / 加分规则：字段存在性、运算符白名单、加分上限"""
    veto = (cfg.get("rules") or {}).get("veto") or {}
    bonus = (cfg.get("rules") or {}).get("bonus") or {}
    field_map = cfg["_field_map"]

    if veto.get("enabled"):
        conds = veto_conditions(cfg)
        if not conds:
            raise ValueError(f"{path} rules.veto 已启用，但 any 条件为空")
        seen = []
        for idx, cond in enumerate(conds):
            name = str(cond.get("field", "")).strip()
            if not name:
                raise ValueError(f"{path} rules.veto.any[{idx}] 存在空条件字段")
            if name in field_map:
                raise ValueError(f"{path} 否决字段 {name} 不应同时作为入模候选字段")
            op = str(cond.get("op", "")).strip()
            if op not in VALID_RULE_OPS:
                raise ValueError(
                    f"{path} rules.veto.any[{idx}] 不支持的运算符 {op!r}，"
                    f"仅支持 {sorted(VALID_RULE_OPS)}"
                )
            if "value" not in cond:
                raise ValueError(f"{path} rules.veto.any[{idx}] 条件缺少 value")
            if isinstance(cond["value"], str) and op not in ("=", "!="):
                raise ValueError(
                    f"{path} rules.veto.any[{idx}] 字符串条件仅支持 = 或 !=，实际为 {op!r}"
                )
            key = (name, op, str(cond["value"]).strip())
            if key in seen:
                raise ValueError(f"{path} rules.veto.any 存在重复条件：{name} {op} {cond['value']}")
            seen.append(key)

    if bonus.get("enabled"):
        max_total = float(bonus.get("max_total", 20))
        if max_total < 0:
            raise ValueError(f"{path} rules.bonus.max_total 不能为负数")
        for block, value_key in (("score_rules", "points"), ("percent_rules", "percent")):
            for idx, rule in enumerate(bonus.get(block) or []):
                all_conds = rule.get("all") or []
                any_conds = rule.get("any") or []
                if not all_conds and not any_conds:
                    raise ValueError(f"{path} rules.bonus.{block}[{idx}] 缺少 all 或 any 条件")
                value = float(rule.get(value_key, 0))
                if value < 0:
                    raise ValueError(f"{path} rules.bonus.{block}[{idx}].{value_key} 不能为负数")
                for cond in list(all_conds) + list(any_conds):
                    name = str(cond.get("field", "")).strip()
                    if not name:
                        raise ValueError(f"{path} rules.bonus.{block}[{idx}] 存在空条件字段")
                    if name not in field_map and name not in veto_condition_fields(cfg):
                        raise ValueError(
                            f"{path} rules.bonus.{block}[{idx}] 条件字段 {name} 不在字段清单或否决字段中"
                        )
                    op = str(cond.get("op", "")).strip()
                    if op not in VALID_RULE_OPS:
                        raise ValueError(
                            f"{path} rules.bonus.{block}[{idx}] 不支持的运算符 {op!r}，"
                            f"仅支持 {sorted(VALID_RULE_OPS)}"
                        )
                    if "value" not in cond:
                        raise ValueError(f"{path} rules.bonus.{block}[{idx}] 条件缺少 value")


# ============================ 否决规则 ============================
def veto_config(cfg):
    """否决规则配置（未配置时返回空字典）"""
    return (cfg.get("rules") or {}).get("veto") or {}


def veto_conditions(cfg):
    """否决条件列表：任一条件命中即 E；未启用时返回空列表"""
    veto = veto_config(cfg)
    if not veto.get("enabled"):
        return []
    return [dict(c) for c in (veto.get("any") or []) if isinstance(c, dict)]


def veto_condition_fields(cfg):
    """否决条件涉及的字段（去重）"""
    return list(dict.fromkeys([
        str(c.get("field", "")).strip() for c in veto_conditions(cfg)
        if str(c.get("field", "")).strip()
    ]))


def veto_condition_text(cfg):
    """否决条件文本，用于 SQL 注释与评估报告"""
    parts = []
    for cond in veto_conditions(cfg):
        field = str(cond.get("field", "")).strip()
        op = str(cond.get("op", "")).strip()
        value = cond.get("value")
        text = f"{field} {op} '{value}'" if isinstance(value, str) else f"{field} {op} {value}"
        if op == "!=":
            text += "（缺失也命中）"
        parts.append(text)
    return " 或 ".join(parts) or "未启用"


def veto_mask(df, cfg):
    """按否决条件生成命中标记：任一条件命中即为 True；!= 条件缺失也命中"""
    conditions = veto_conditions(cfg)
    if not conditions:
        return np.zeros(len(df), dtype=bool)
    miss = [f for f in veto_condition_fields(cfg) if f not in df.columns]
    if miss:
        raise ValueError(f"数据中缺少否决规则字段：{miss}")

    mask = np.zeros(len(df), dtype=bool)
    for cond in conditions:
        field = str(cond["field"]).strip()
        op = str(cond["op"]).strip()
        value = cond["value"]
        if isinstance(value, str):
            series = df[field].astype("string").str.strip()
            hit = (series == value).fillna(False) if op == "=" else (series != value).fillna(True)
        else:
            series = pd.to_numeric(df[field], errors="coerce")
            rhs = float(value)
            ops = {
                ">": lambda s, v: (s > v).fillna(False),
                ">=": lambda s, v: (s >= v).fillna(False),
                "<": lambda s, v: (s < v).fillna(False),
                "<=": lambda s, v: (s <= v).fillna(False),
                "=": lambda s, v: (s == v).fillna(False),
                "!=": lambda s, v: (s != v).fillna(True),
            }
            hit = ops[op](series, rhs)
        mask |= hit.to_numpy(dtype=bool)
    return mask


# ============================ 加分规则 ============================
def bonus_condition_fields(cfg):
    """加分规则条件涉及的字段；未启用时返回空列表"""
    bonus = (cfg.get("rules") or {}).get("bonus") or {}
    if not bonus.get("enabled"):
        return []
    names = []
    for block in ("score_rules", "percent_rules"):
        for rule in bonus.get(block) or []:
            for key in ("all", "any"):
                for cond in rule.get(key) or []:
                    names.append(str(cond.get("field", "")).strip())
    return list(dict.fromkeys([n for n in names if n]))


def rule_fields(cfg):
    """训练 / 评估需要额外读取的业务规则字段（否决条件字段 + 加分条件字段）"""
    return list(dict.fromkeys(veto_condition_fields(cfg) + bonus_condition_fields(cfg)))


def rule_condition_mask(df, condition):
    """执行加分规则的条件（支持 > >= < <= = !=，缺失不命中）"""
    field = str(condition["field"]).strip()
    op = str(condition["op"]).strip()
    value = condition["value"]
    if isinstance(value, str):
        series = df[field].astype("string").str.strip()
        rhs = value
    else:
        series = pd.to_numeric(df[field], errors="coerce")
        rhs = float(value)

    ops = {
        ">": lambda s, v: s > v,
        ">=": lambda s, v: s >= v,
        "<": lambda s, v: s < v,
        "<=": lambda s, v: s <= v,
        "=": lambda s, v: s == v,
        "!=": lambda s, v: s != v,
    }
    if op not in ops:
        raise ValueError(f"不支持的规则运算符：{op}")
    mask = ops[op](series, rhs)
    return (mask & series.notna()).to_numpy(dtype=bool)


def rule_mask(df, rule):
    """规则命中：all 条件全部满足，且 any 条件至少满足一个"""
    mask = np.ones(len(df), dtype=bool)
    for condition in rule.get("all") or []:
        mask &= rule_condition_mask(df, condition)
    any_conditions = rule.get("any") or []
    if any_conditions:
        any_mask = np.zeros(len(df), dtype=bool)
        for condition in any_conditions:
            any_mask |= rule_condition_mask(df, condition)
        mask &= any_mask
    return mask
