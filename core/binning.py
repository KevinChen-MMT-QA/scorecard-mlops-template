# core/binning.py
"""卡方分箱、WOE / IV 计算与本地评分。

分箱 / WOE 口径由训练、评估共用，保证 SQL 出分与训练一致；缺失值单独成箱。
"""
import numpy as np
import pandas as pd

from core.config import MISSING_LABEL, valid_flag_values


def fit_bin_rule(values, target, cfg, field):
    """对一个字段做卡方分箱并计算 WOE/IV；缺失值单独成一箱"""
    prep_num = (cfg.get("preprocessing") or {}).get("numeric") or {}
    max_initial = int(prep_num.get("max_initial_bins", 10))
    max_bin = int(prep_num.get("max_bin", 5))
    min_pct = float(prep_num.get("min_bin_pct", 0.05))
    y = np.asarray(target, dtype=float)
    ftype = field["feature_type"]

    if ftype == "numeric":
        vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(vals)
        edges = _numeric_edges(vals[valid], y[valid], max_initial, max_bin, min_pct)
        bins = [
            {"label": _interval_label(edges[i], edges[i + 1]), "kind": "numeric",
             "low": edges[i], "high": edges[i + 1]}
            for i in range(len(edges) - 1)
        ]
        labels = np.full(len(vals), MISSING_LABEL, dtype=object)
        if valid.any():
            idx = _numeric_index(vals[valid], edges)
            labels[valid] = [bins[i]["label"] for i in idx]
        rule = {"field": field["name"], "feature_type": ftype, "type": "numeric",
                "edges": edges, "bins": bins,
                "invalid_values": list(field.get("invalid_values") or [])}
    else:
        series = values.astype("string")
        cats = sorted(pd.unique(series.dropna().astype(str)).tolist()) if len(series) else []
        bins = [{"label": str(c), "kind": "category", "value": str(c)} for c in cats]
        labels = series.fillna(MISSING_LABEL).astype(object).to_numpy()
        rule = {"field": field["name"], "feature_type": ftype, "type": "category",
                "values": [str(c) for c in cats], "bins": bins,
                "invalid_values": [str(v) for v in (field.get("invalid_values") or [])]}
        if ftype == "flag":
            rule["valid_values"] = valid_flag_values(cfg)

    stats = _woe_stats(labels, [b["label"] for b in bins], y)
    rule["missing"] = stats["missing"]
    for bin_cfg, bin_stat in zip(bins, stats["bins"]):
        bin_cfg.update(bin_stat)
    rule["iv"] = stats["iv"]
    return rule


def _numeric_edges(vals, y, max_initial, max_bin, min_pct):
    """卡方分箱主流程：初始分位分箱 -> 合并小样本箱 -> 卡方合并至 max_bin"""
    unique = np.unique(vals)
    if len(unique) <= 1:
        return [float("-inf"), float("inf")]
    q = min(max_initial, len(unique))
    quantile_bins = pd.qcut(pd.Series(vals), q=q, duplicates="drop")
    rights = [round(float(iv.right), 6) for iv in quantile_bins.cat.categories]
    edges = sorted(set([float("-inf")] + rights))
    edges[-1] = float("inf")
    if len(edges) <= 2:
        return [float("-inf"), float("inf")]

    while len(edges) - 1 > 1:
        total, bad = _bin_counts(edges, vals, y)
        k = len(total)
        small = [i for i in range(k) if total[i] < min_pct * len(vals)]
        if k <= max_bin and not small:
            break
        chi = [_chi2(total[i], bad[i], total[i + 1], bad[i + 1]) for i in range(k - 1)]
        if small:
            i = min(small, key=lambda j: total[j])
            if i == 0:
                pos = 0
            elif i == k - 1:
                pos = k - 2
            else:
                pos = i - 1 if chi[i - 1] <= chi[i] else i
        else:
            pos = int(np.argmin(chi))
        edges = edges[:pos + 1] + edges[pos + 2:]

    # 收尾：坏率完全一致的相邻箱合并，避免 SQL 中出现重复分值分支
    while len(edges) - 1 > 1:
        total, bad = _bin_counts(edges, vals, y)
        rates = np.divide(bad, total, out=np.zeros_like(bad), where=total > 0)
        pos = next((i for i in range(len(rates) - 1) if abs(rates[i] - rates[i + 1]) < 1e-9), None)
        if pos is None:
            break
        edges = edges[:pos + 1] + edges[pos + 2:]
    return edges


def _bin_counts(edges, vals, y):
    idx = _numeric_index(vals, edges)
    k = len(edges) - 1
    total = np.bincount(idx, minlength=k).astype(float)
    bad = np.bincount(idx, weights=y, minlength=k).astype(float)
    return total, bad


def _chi2(n1, b1, n2, b2):
    g1, g2 = n1 - b1, n2 - b2
    total, bad, good = n1 + n2, b1 + b2, g1 + g2
    if total <= 0 or bad <= 0 or good <= 0:
        return float("inf")
    expected = [bad * n1 / total, good * n1 / total, bad * n2 / total, good * n2 / total]
    observed = [b1, g1, b2, g2]
    return float(sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0))


def _numeric_index(vals, edges):
    """按右闭区间定位分箱下标：(edges[i], edges[i+1]]"""
    idx = np.searchsorted(np.asarray(edges, dtype=float), np.asarray(vals, dtype=float), side="left") - 1
    return np.clip(idx, 0, len(edges) - 2).astype(int)


def _woe_stats(labels, bin_labels, y):
    """按分箱统计样本、坏样本、WOE 与 IV（拉普拉斯平滑，缺失单独一箱）"""
    total_bad = float(y.sum())
    total_good = float(len(y) - total_bad)
    all_labels = list(bin_labels) + [MISSING_LABEL]
    active_k = max(sum(int((labels == lab).sum()) > 0 for lab in all_labels), 1)

    def _one(label):
        mask = labels == label
        count = int(mask.sum())
        bad = float(y[mask].sum())
        good = count - bad
        if count == 0:
            woe, iv = 0.0, 0.0
        else:
            bad_pct = (bad + 0.5) / (total_bad + 0.5 * active_k)
            good_pct = (good + 0.5) / (total_good + 0.5 * active_k)
            woe = float(np.log(bad_pct / good_pct))
            iv = float((bad_pct - good_pct) * woe)
        return {"label": label, "count": count, "bad": int(bad), "good": int(good),
                "bad_rate": round(bad / count, 6) if count else 0.0,
                "woe": round(woe, 6), "iv": round(iv, 6)}

    bins = [_one(lab) for lab in bin_labels]
    missing = _one(MISSING_LABEL)
    iv_sum = sum(b["iv"] for b in bins) + missing["iv"]
    return {"bins": bins, "missing": missing, "iv": round(float(iv_sum), 6)}


def _interval_label(low, high):
    return f"({_fmt_num(low)}, {_fmt_num(high)}]"


def _fmt_num(value):
    if np.isneginf(value):
        return "-inf"
    if np.isposinf(value):
        return "+inf"
    return f"{value:.6g}"


def bin_labels_for_rule(rule, values):
    """按规则给原始值打上分箱标签（训练集重算分数、校验 SQL 时使用）"""
    if rule["type"] == "numeric":
        vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        labels = np.full(len(vals), MISSING_LABEL, dtype=object)
        valid = ~np.isnan(vals)
        if valid.any():
            idx = _numeric_index(vals[valid], rule["edges"])
            labels[valid] = [rule["bins"][i]["label"] for i in idx]
        return pd.Series(labels, index=values.index)
    series = values.astype("string").str.strip().fillna(MISSING_LABEL).astype(object)
    known = set(rule["values"]) | {MISSING_LABEL}
    return pd.Series(series.where(series.isin(known), "OTHER").to_numpy(), index=values.index)


def woe_map_for_rule(rule):
    mapping = {b["label"]: b["woe"] for b in rule["bins"]}
    mapping[MISSING_LABEL] = rule["missing"]["woe"]
    return mapping


def score_with_rules(df, rules, base_component):
    """本地按最终规则累加分值（训练集分位阈值用；线上以 SQL 为准）"""
    score = pd.Series(float(base_component), index=df.index, dtype=float)
    for rule in rules:
        points = {b["label"]: b["points"] for b in rule["bins"]}
        points[MISSING_LABEL] = rule["missing"]["points"]
        labels = bin_labels_for_rule(rule, df[rule["field"]])
        score = score + labels.map(points).fillna(0.0).astype(float)
    return score
