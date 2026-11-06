# core/reporting.py
"""报告产出：训练明细 CSV 组装与评估 Markdown 报告渲染。"""
import os

import numpy as np
import pandas as pd

from core.rules import veto_condition_text, veto_config


def df_to_markdown(df):
    """DataFrame 转 markdown（不依赖 tabulate）"""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(_fmt_cell(v) for v in row) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


def _fmt_cell(value):
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def build_score_detail(cfg, rules_all, coef, final_fields, final_status):
    """字段-分箱-WOE-系数-分值明细"""
    detail_rows = []
    for name, rule in rules_all.items():
        is_final = name in final_fields
        coefficient = coef.get(name, 0.0) if is_final else 0.0
        rows = [("MISSING（缺失）", rule["missing"])] + [(b["label"], b) for b in rule["bins"]]
        for label, bin_stat in rows:
            points = bin_stat.get("points", 0.0) if is_final else 0.0
            detail_rows.append({
                "字段名": name,
                "中文名": cfg["_field_map"][name].get("cn_name", ""),
                "特征类型": rule["feature_type"],
                "分箱": label,
                "样本数": bin_stat["count"],
                "坏样本数": bin_stat["bad"],
                "坏客户率": bin_stat["bad_rate"],
                "WOE": bin_stat["woe"],
                "模型系数": round(coefficient, 6),
                "分值贡献": round(points, 6),
                "字段状态": final_status[name],
            })
    return pd.DataFrame(detail_rows)


def enrich_feature_quality(quality, rules_all, vif_values, final_status, status):
    """特征质量表补充 IV / VIF / 最终筛选状态"""
    quality["IV"] = quality["字段名"].map(lambda x: rules_all[x]["iv"] if x in rules_all else np.nan)
    quality["VIF"] = quality["字段名"].map(lambda x: round(vif_values[x], 4) if x in vif_values else np.nan)
    quality["最终状态"] = quality["字段名"].map(lambda x: final_status.get(x, status.get(x, "")))
    return quality


def top_field_summary(detail_path, top_n=10):
    """入模字段分值跨度 TopN（读取 score_detail.csv）"""
    if not os.path.exists(detail_path):
        return ""
    detail = pd.read_csv(detail_path)
    used = detail[detail["字段状态"] == "入模"]
    summary = (used.groupby(["字段名", "中文名"])["分值贡献"]
               .apply(lambda s: s.max() - s.min()).reset_index()
               .rename(columns={"分值贡献": "分值跨度"}))
    summary = summary.sort_values("分值跨度", ascending=False).head(top_n)
    return df_to_markdown(summary)


def build_model_report(cfg, eval_result, abcd, thresholds, train_cnt, test_cnt, field_count,
                       top_fields="", pk="unt_scl_cdt_cd"):
    """模型交付说明（Markdown）"""
    veto = veto_config(cfg)
    bonus = (cfg.get("rules") or {}).get("bonus") or {}
    veto_text = veto_condition_text(cfg)
    bonus_text = (
        f"分值类 {len(bonus.get('score_rules') or [])} 条、百分比类 "
        f"{len(bonus.get('percent_rules') or [])} 条，合计上限 {float(bonus.get('max_total', 20)):.2f} 分"
        if bonus.get("enabled") else "未启用"
    )
    threshold_text = "，".join([f"{lab} ≥ {th:.2f}" for th, lab in thresholds])

    return f"""# 评分卡模型说明（{cfg['scenario']}）

## 基础信息
- 基准分：{cfg['model'].get('base_score')}，PDO：{cfg['model'].get('pdo')}
- 训练样本量：{train_cnt}，测试样本量：{test_cnt}
- 入模字段数：{field_count}（明细见 `report/score_detail.csv`）
- 场景配置：`scenario.yaml`（表结构变更仅需修改该文件与 `fields.csv`）
- 线上产物：`report/scorecard.sql`（Hive 单条 SELECT，无 pkl、无推理代码）

## 业务规则
- 否决规则（{"启用" if veto.get("enabled") else "未启用"}）：{veto_text} → 等级 E，分数照常输出
- 训练集剔除：{"是" if veto.get("exclude_from_train") else "否"}（测试集保留 E 客户用于评估）
- 加分规则（{bonus_text}）：基础分负分归零后，先加分值类，再按（归零分 + 分值类加分）叠加百分比类；ABCD 阈值按基础分计算

## 模型效果（SQL 执行结果，AUC/KS/PSI 不含 E 客户）
{df_to_markdown(eval_result)}

## A/B/C/D/E 客户分层（测试集）
- SQL 内置 A/B/C/D 阈值：{threshold_text}
- E 为否决规则命中等级，分数仍保留
{df_to_markdown(abcd)}

## 入模字段分值跨度 Top10
{top_fields if top_fields else '（详见 score_detail.csv）'}

## SQL 使用说明
1. 分区参数：`${{bizdate}}`，执行示例：`hive -hivevar bizdate=20261031 -f report/scorecard.sql`
2. 取数口径：按 `part_dt` 分区取当日快照（分区日期即快照日期，不做去重）
3. 输出列：`{pk}`（唯一主键，原样输出）、`score`、`customer_level`（分数越高客户越优）
4. 缺失值：训练时单独成箱，SQL 中同样单独给分；未见类别按 WOE=0 不贡献分

## 监控建议
- 月度监控分数 PSI、各分层坏客户率、入模字段分布漂移
- 特征缺失率变化超过 10pct 时，重新评估 `{(cfg.get('preprocessing') or {}).get('missing_rate_threshold', 0.8)}` 阈值下的字段可用性
"""
