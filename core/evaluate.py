# core/evaluate.py
"""评分卡评估编排：直接执行场景 report/scorecard.sql（本地 pandas 等价执行器）并计算指标。

产物（输出目录由场景配置 sql.report_dir 指定）：
    eval_result.csv    AUC/KS/PSI 等模型效果指标（不含 E 客户）
    abcd_rule.csv      A/B/C/D/E 分层坏率与分数区间
    model_report.md    模型交付说明
"""
import os
import warnings

import pandas as pd
from sklearn.metrics import roc_auc_score

from core.config import load_config
from core.evaluator import abcd_table, calc_ks, execute_scorecard_sql, parse_scorecard_sql, psi_calc
from core.preprocess import prepare_dataset
from core.reporting import build_model_report, top_field_summary

warnings.filterwarnings("ignore")


def main(config_path, bizdate=None):
    """评估入口：config_path 为场景 YAML，与训练使用同一分区以保证口径一致"""
    cfg = load_config(config_path)
    report_dir = cfg["sql"]["report_dir"]
    os.makedirs(report_dir, exist_ok=True)
    print(f"场景：{cfg['scenario']} | 配置：{os.path.abspath(config_path)}")
    df, _, _ = prepare_dataset(cfg, bizdate=bizdate)
    pk = cfg["source"]["primary_key"]
    sql_path = cfg["sql"].get("output_file", os.path.join(report_dir, "scorecard.sql"))
    with open(sql_path, encoding="utf-8") as f:
        sql_text = f.read()

    print(f"已读取评分卡 SQL：{sql_path}")
    print("本地未安装 Spark/DuckDB，使用内置 pandas 等价执行器评估 SQL 结果")
    pred = execute_scorecard_sql(sql_text, df, cfg=cfg, pk=pk)
    merged = df[[pk, "y", "split_tag"]].merge(pred, on=pk, how="inner")
    train = merged[merged["split_tag"] == "train"]
    test = merged[merged["split_tag"] == "test"]
    # 否决客户（E）不参与模型区分度指标，单独在 ABCD 表中统计
    train_model = train[train["customer_level"] != "E"]
    test_model = test[test["customer_level"] != "E"]

    # 分数越高客户越好，因此用 -score 作为违约概率的单调替身计算 AUC / KS
    auc_train = float(roc_auc_score(train_model["y"], -train_model["score"]))
    auc_test = float(roc_auc_score(test_model["y"], -test_model["score"]))
    ks_train = calc_ks(train_model["y"], -train_model["score"])
    ks_test = calc_ks(test_model["y"], -test_model["score"])
    psi_val = psi_calc(train_model["score"], test_model["score"], bins=10)

    veto_test = int((test["customer_level"] == "E").sum())
    eval_result = pd.DataFrame({
        "metric": ["AUC_train", "AUC_test", "KS_train", "KS_test", "PSI(train vs test)",
                   "train_cnt", "test_cnt", "veto_cnt_test", "veto_rate_test", "scorecard_fields"],
        "value": [round(auc_train, 6), round(auc_test, 6), round(ks_train, 6), round(ks_test, 6),
                  round(psi_val, 6), len(train), len(test), veto_test,
                  round(veto_test / len(test), 6) if len(test) else 0.0,
                  len(parse_scorecard_sql(sql_text)["blocks"])],
    })
    eval_result.to_csv(os.path.join(report_dir, "eval_result.csv"), index=False, encoding="utf-8-sig")

    abcd = abcd_table(test)
    abcd.to_csv(os.path.join(report_dir, "abcd_rule.csv"), index=False, encoding="utf-8-sig")

    print("\n评估指标：")
    print(eval_result.to_string(index=False))
    print("\nA/B/C/D/E 客户分层（测试集，阈值已写入 SQL）：")
    print(abcd.to_string(index=False))

    # 模型说明文档
    parsed = parse_scorecard_sql(sql_text)
    top_fields = top_field_summary(os.path.join(report_dir, "score_detail.csv"))
    report_content = build_model_report(
        cfg, eval_result, abcd,
        thresholds=parsed["levels"],
        train_cnt=len(train),
        test_cnt=len(test),
        field_count=len(parsed["blocks"]),
        top_fields=top_fields,
        pk=pk,
    )
    report_path = os.path.join(report_dir, "model_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n评估报告已生成：{report_path}")
