#!/usr/bin/env python
"""评分卡模板统一入口。

用法：
    python run.py list
    python run.py init --scenario auto_loan
    python run.py train    --scenario industry_finance --bizdate 20261031
    python run.py evaluate --scenario industry_finance --bizdate 20261031
    python run.py pipeline --scenario industry_finance --bizdate 20261031
"""
import argparse
import os
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR = os.path.join(ROOT_DIR, "scenarios")
TEMPLATE_DIR = os.path.join(SCENARIO_DIR, "_template")


def scenario_config_path(args):
    """解析场景配置路径：优先 --config，其次 scenarios/<name>/scenario.yaml"""
    if getattr(args, "config", None):
        path = os.path.abspath(args.config)
        if not os.path.exists(path):
            raise SystemExit("配置文件不存在：%s" % path)
        return path
    name = getattr(args, "scenario", None)
    if not name:
        raise SystemExit("请指定 --scenario <name> 或 --config <scenario.yaml>")
    path = os.path.join(SCENARIO_DIR, name, "scenario.yaml")
    if not os.path.exists(path):
        raise SystemExit("场景不存在或缺少 scenario.yaml：%s" % path)
    return path


def cmd_list(args):
    """列出已注册场景"""
    if not os.path.isdir(SCENARIO_DIR):
        print("scenarios 目录不存在")
        return
    names = sorted(
        name for name in os.listdir(SCENARIO_DIR)
        if os.path.isdir(os.path.join(SCENARIO_DIR, name)) and not name.startswith("_")
    )
    if not names:
        print("暂无场景，可用 python run.py init --scenario <name> 创建")
        return
    print("可用场景：")
    for name in names:
        config = os.path.join(SCENARIO_DIR, name, "scenario.yaml")
        mark = "OK" if os.path.exists(config) else "缺少 scenario.yaml"
        print("  %-24s %s" % (name, mark))


def cmd_init(args):
    """从 _template 复制新场景"""
    if not os.path.isdir(TEMPLATE_DIR):
        raise SystemExit("模板目录不存在：%s" % TEMPLATE_DIR)
    target = os.path.join(SCENARIO_DIR, args.scenario)
    if os.path.exists(target):
        raise SystemExit("场景已存在：%s" % target)
    shutil.copytree(TEMPLATE_DIR, target)
    config_path = os.path.join(target, "scenario.yaml")
    with open(config_path, encoding="utf-8") as f:
        content = f.read()
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content.replace("__SCENARIO__", args.scenario))
    print("场景已创建：%s" % target)
    print("请补充 fields.csv、samples/ 正负样本与 scenario.yaml 源表配置")


def cmd_train(args):
    from core.train import main as train_main
    train_main(scenario_config_path(args), bizdate=args.bizdate)


def cmd_evaluate(args):
    from core.evaluate import main as evaluate_main
    evaluate_main(scenario_config_path(args), bizdate=args.bizdate)


def cmd_pipeline(args):
    config_path = scenario_config_path(args)
    from core.train import main as train_main
    from core.evaluate import main as evaluate_main
    train_main(config_path, bizdate=args.bizdate)
    evaluate_main(config_path, bizdate=args.bizdate)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="评分卡模板统一入口：场景化训练、评估与报告生成",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出已注册场景").set_defaults(func=cmd_list)

    p_init = sub.add_parser("init", help="从 _template 创建新场景")
    p_init.add_argument("--scenario", required=True, help="新场景名称（即目录名）")
    p_init.set_defaults(func=cmd_init)

    for name, func, text in (
        ("train", cmd_train, "训练并生成评分卡 SQL 与特征明细"),
        ("evaluate", cmd_evaluate, "执行评分卡 SQL 并生成评估报告"),
        ("pipeline", cmd_pipeline, "训练 + 评估一次完成"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--scenario", help="场景名称，对应 scenarios/<name>/scenario.yaml")
        p.add_argument("--config", help="直接指定 scenario.yaml 路径，优先于 --scenario")
        p.add_argument("--bizdate", help="Hive 分区日期（yyyymmdd），spark 模式必传")
        p.set_defaults(func=func)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
