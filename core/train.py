# core/train.py
"""评分卡训练编排：配置驱动分箱 / WOE / IV / VIF / 逻辑回归，直接产出可运行的评分卡 SQL。

产物（不落 pkl、不产出推理代码），输出目录由场景配置 sql.report_dir 指定：
    scorecard.sql        可直接在 Hive 执行的单条 SELECT 评分卡 SQL
    score_detail.csv     字段-分箱-WOE-系数-分值明细
    feature_quality.csv  特征质量与筛选漏斗（缺失率/IV/VIF/最终状态）
"""
import os
import warnings

from core.binning import fit_bin_rule
from core.config import feature_fields, load_config
from core.modeling import (
    assign_points,
    build_woe_frame,
    calc_vif_frame,
    compute_thresholds,
    fit_logistic_model,
)
from core.preprocess import prepare_dataset
from core.reporting import build_score_detail, enrich_feature_quality
from core.sql_builder import build_scorecard_sql

warnings.filterwarnings("ignore")


def main(config_path, bizdate=None):
    """训练入口：config_path 为场景 YAML，输出目录取自 sql.report_dir"""
    cfg = load_config(config_path)
    report_dir = cfg["sql"]["report_dir"]
    os.makedirs(report_dir, exist_ok=True)
    print(f"场景：{cfg['scenario']} | 配置：{os.path.abspath(config_path)}")
    df, quality, status = prepare_dataset(cfg, bizdate=bizdate)
    train_df = df[df["split_tag"] == "train"].reset_index(drop=True)
    y_train = train_df["y"].to_numpy()
    print(f"样本总数 {len(df)}，训练集 {len(train_df)}，测试集 {len(df) - len(train_df)}")

    # 1. 可用字段 -> 卡方分箱 / WOE / IV
    usable = [f for f in feature_fields(cfg) if status.get(f["name"]) == "可用"]
    print(f"预处理后可用字段 {len(usable)} 个")
    rules_all = {}
    for field in usable:
        rules_all[field["name"]] = fit_bin_rule(train_df[field["name"]], y_train, cfg, field)

    # 2. IV 筛选
    iv_thr = float(cfg["model"].get("iv_threshold", 0.02))
    iv_keep = [n for n, r in rules_all.items() if r["iv"] > iv_thr and len(r["bins"]) > 1]
    print(f"IV>{iv_thr} 字段 {len(iv_keep)} 个")

    # 3. VIF 筛选
    vif_thr = float(cfg["model"].get("vif_threshold", 5.0))
    woe_train = build_woe_frame(train_df, [rules_all[n] for n in iv_keep])
    vif_keep, vif_values = calc_vif_frame(woe_train, iv_keep, vif_thr)
    print(f"VIF<={vif_thr} 字段 {len(vif_keep)} 个：{vif_keep}")

    if not vif_keep:
        raise RuntimeError("所有候选字段均被 IV/VIF 筛除，请检查场景配置或样本数据")

    # 4. 逻辑回归与分值换算
    coef, final_fields, factor, base_component = fit_logistic_model(woe_train, vif_keep, y_train, cfg)
    print(f"系数非 0 入模字段 {len(final_fields)} 个")

    final_rules = [rules_all[n] for n in final_fields]
    assign_points(final_rules, coef, factor)

    # 5. 训练集基础分分位点作为 A/B/C/D 阈值
    thresholds = compute_thresholds(train_df, final_rules, base_component)

    # 6. 生成 SQL
    sql_path = cfg["sql"].get("output_file", os.path.join(report_dir, "scorecard.sql"))
    os.makedirs(os.path.dirname(sql_path) or ".", exist_ok=True)
    with open(sql_path, "w", encoding="utf-8") as f:
        f.write(build_scorecard_sql(cfg, final_rules, base_component, thresholds))
    print(f"评分卡 SQL 已生成：{sql_path}")

    # 7. 字段明细
    final_status = dict(status)
    for name in rules_all:
        if status.get(name) != "可用":
            continue
        if name not in iv_keep:
            final_status[name] = "IV不足剔除"
        elif name not in vif_keep:
            final_status[name] = "共线性剔除"
        elif name not in final_fields:
            final_status[name] = "系数为0剔除"
        else:
            final_status[name] = "入模"

    detail = build_score_detail(cfg, rules_all, coef, final_fields, final_status)
    detail.to_csv(os.path.join(report_dir, "score_detail.csv"), index=False, encoding="utf-8-sig")

    # 8. 特征质量与筛选漏斗
    quality = enrich_feature_quality(quality, rules_all, vif_values, final_status, status)
    quality.to_csv(os.path.join(report_dir, "feature_quality.csv"), index=False, encoding="utf-8-sig")

    t_a, t_b, t_c = thresholds
    print("\n筛选漏斗：")
    print(f"  字段清单 {len(cfg['fields'])} 个 -> "
          f"预处理可用 {len(usable)} -> IV保留 {len(iv_keep)} -> VIF保留 {len(vif_keep)} -> "
          f"系数非0入模 {len(final_fields)}")
    print(f"  基准分项 {base_component:.4f}，A/B/C/D 阈值（按基础分） A>={t_a} B>={t_b} C>={t_c} D<{t_c}")
    print(f"  明细已输出：{os.path.join(report_dir, 'score_detail.csv')}、"
          f"{os.path.join(report_dir, 'feature_quality.csv')}")
