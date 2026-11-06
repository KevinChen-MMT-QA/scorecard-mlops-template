# core/sql_builder.py
"""评分卡 SQL 生成：分箱分值内联、否决标记、负分归零、两级加分与 A/B/C/D/E 分级。

生成的是 Hive 可直接执行的单条 SELECT，SQL 即线上模型产物。
"""
from core.rules import rule_fields, veto_condition_text, veto_conditions


def fmt_sql_number(value):
    """SQL 中的数值格式：源表精度最高 decimal(18,6)，统一保留 6 位小数"""
    return f"{float(value):.6f}"


def sql_case_block(rule):
    """单个字段的分值 CASE WHEN（分箱分值内联）"""
    field = rule["field"]
    points_missing = fmt_sql_number(rule["missing"]["points"])
    lines = ["        CASE"]
    if rule["type"] == "numeric":
        conds = [f"{field} IS NULL"]
        invalid = [fmt_sql_number(v) for v in rule["invalid_values"]]
        if invalid:
            conds.append(f"{field} IN ({', '.join(invalid)})")
        lines.append(f"            WHEN {' OR '.join(conds)} THEN {points_missing}")
        for bin_cfg in rule["bins"][:-1]:
            lines.append(f"            WHEN {field} <= {fmt_sql_number(bin_cfg['high'])} THEN "
                         f"{fmt_sql_number(bin_cfg['points'])}")
        lines.append(f"            ELSE {fmt_sql_number(rule['bins'][-1]['points'])}")
    else:
        if rule["feature_type"] == "flag":
            allowed = ", ".join(f"'{str(v).replace(chr(39), chr(39) * 2)}'"
                                for v in rule["valid_values"])
            lines.append(f"            WHEN {field} IS NULL OR TRIM({field}) = '' "
                         f"OR TRIM({field}) NOT IN ({allowed}) THEN {points_missing}")
        else:
            lines.append(f"            WHEN {field} IS NULL OR TRIM({field}) = '' THEN {points_missing}")
        for bin_cfg in rule["bins"]:
            value = str(bin_cfg["value"]).replace("'", "''")
            lines.append(f"            WHEN TRIM({field}) IN ('{value}') THEN "
                         f"{fmt_sql_number(bin_cfg['points'])}")
        # 训练集未出现的类别：按 WOE=0 处理，不贡献分
        lines.append("            ELSE 0.000000")
    lines.append(f"        END AS sc_{field}")
    return "\n".join(lines)


def sql_condition(cond):
    """加分条件转 SQL：字段 op 值，字符串自动加引号"""
    field = str(cond["field"]).strip()
    op = str(cond["op"]).strip()
    value = cond["value"]
    if isinstance(value, str):
        value_sql = "'" + value.replace("'", "''") + "'"
    else:
        value_sql = fmt_sql_number(value)
    return f"{field} {op} {value_sql}"


def sql_rule_conditions(rule):
    """规则条件转 SQL：all 内部 AND，any 内部 OR，两者之间 AND"""
    parts = []
    all_conds = rule.get("all") or []
    any_conds = rule.get("any") or []
    if all_conds:
        parts.append(" AND ".join(f"({sql_condition(c)})" for c in all_conds))
    if any_conds:
        parts.append("(" + " OR ".join(f"({sql_condition(c)})" for c in any_conds) + ")")
    return " AND ".join(parts)


def sql_sum_case(rules, value_key):
    """多条加分规则命中值累加；无规则时返回 0"""
    if not rules:
        return "0.000000"
    parts = [
        f"(CASE WHEN {sql_rule_conditions(rule)} THEN {fmt_sql_number(rule.get(value_key, 0))} "
        f"ELSE 0.000000 END)"
        for rule in rules
    ]
    return " + ".join(parts)


def sql_veto_condition(cond):
    """单个否决条件转 SQL：字符串按 TRIM 比较，!= 条件缺失也命中"""
    field = str(cond["field"]).strip()
    op = str(cond["op"]).strip()
    value = cond["value"]
    if isinstance(value, str):
        value_sql = "'" + value.replace("'", "''") + "'"
        if op == "=":
            return f"TRIM({field}) = {value_sql}"
        return f"({field} IS NULL OR TRIM({field}) <> {value_sql})"
    value_sql = fmt_sql_number(value)
    if op == "!=":
        return f"({field} IS NULL OR CAST({field} AS DOUBLE) <> {value_sql})"
    return f"CAST({field} AS DOUBLE) {op} {value_sql}"


def sql_veto_expr(cfg):
    """否决命中表达式：任一否决条件命中即为 1"""
    conditions = veto_conditions(cfg)
    if not conditions:
        return "0"
    conds = [sql_veto_condition(c) for c in conditions]
    return "CASE WHEN " + " OR ".join(conds) + " THEN 1 ELSE 0 END"


def build_scorecard_sql(cfg, rules, base_component, thresholds):
    """生成 Hive 可直接执行的评分卡 SQL（否决 + 负分归零 + 两级加分 + 分值内联）"""
    from core.feature_engineering import feature_sql_text

    src = cfg["source"]
    model = cfg["model"]
    pk = src["primary_key"]
    part_col = src["partition_field"]
    table = f"{src.get('database')}.{src['table_name']}" if src.get("database") else src["table_name"]
    score_alias = cfg["sql"].get("score_alias", "score")
    level_alias = cfg["sql"].get("level_alias", "customer_level")
    decimals = int(model.get("score_decimals", 2))
    t_a, t_b, t_c = thresholds

    bonus = (cfg.get("rules") or {}).get("bonus") or {}
    bonus_enabled = bool(bonus.get("enabled"))
    score_rules = (bonus.get("score_rules") or []) if bonus_enabled else []
    percent_rules = (bonus.get("percent_rules") or []) if bonus_enabled else []
    max_total = float(bonus.get("max_total", 20))
    points_expr = sql_sum_case(score_rules, "points")
    percent_expr = sql_sum_case(percent_rules, "percent")
    veto_expr = sql_veto_expr(cfg)

    feature_cols = [r["field"] for r in rules]
    select_cols = list(dict.fromkeys([pk] + feature_cols + rule_fields(cfg)))
    select_sql = ", ".join(select_cols)
    case_sql = ",\n".join(sql_case_block(rule) for rule in rules)
    score_expr = " + ".join([fmt_sql_number(base_component)] + [f"sc_{r['field']}" for r in rules])
    veto_rule_text = veto_condition_text(cfg)

    # 决定 WITH 链的第一个 CTE：特征衍生 or 单表 src
    fe_cfg = cfg.get("feature_engineering") or {}
    if fe_cfg.get("enabled") and fe_cfg.get("sql_file"):
        feature_sql = feature_sql_text(cfg)
        feature_sql_clean = feature_sql.rstrip().rstrip(";").strip()
        engineered_cte = "engineered AS (\n        " + feature_sql_clean + "\n    ),\n"
        src_from = "engineered"
        src_where = ""
        comment_feature = (
            "特征衍生：内嵌 " + fe_cfg["sql_file"] + "（含多表 JOIN + 衍生字段计算）\n"
        )
    else:
        engineered_cte = ""
        src_from = table
        src_where = "\n    WHERE " + part_col + " = '${bizdate}'"
        comment_feature = ""

    sql_text = (
        "-- 评分卡 SQL（场景：" + str(cfg.get("scenario")) + "，由 core/train.py 自动生成，请勿手工修改）\n"
        "-- 基准分：" + str(model.get("base_score")) + "，PDO：" + str(model.get("pdo"))
        + "；入模字段：" + str(len(rules)) + " 个\n"
        "-- 主键：" + pk + "（唯一标识，原样输出，不做别名映射）\n"
        "-- 否决规则：" + veto_rule_text + " → 等级 E，分数仍照常输出\n"
        "-- 加分规则：基础分负分归零后，先加分值类，再按（归零分 + 分值类加分）叠加百分比类\n"
        "-- 加分上限：" + format(max_total, ".2f") + " 分；ABCD 阈值按基础分计算，加分后再分级\n"
        "-- " + comment_feature + "-- 参数：${bizdate} 为分区日期，执行示例：hive -hivevar bizdate=20261031 -f scorecard.sql\n\n"
        "WITH " + engineered_cte + "src AS (\n"
        "    SELECT " + select_sql + "\n"
        "    FROM " + src_from + src_where + "\n"
        "),\n"
        "base AS (\n"
        "    SELECT\n"
        "        " + pk + ",\n"
        + case_sql + ",\n"
        "        " + veto_expr + " AS is_veto,\n"
        "        " + points_expr + " AS bonus_points,\n"
        "        " + percent_expr + " AS bonus_pct\n"
        "    FROM src\n"
        "),\n"
        "scored AS (\n"
        "    SELECT\n"
        "        " + pk + ",\n"
        "        is_veto,\n"
        "        bonus_points,\n"
        "        bonus_pct,\n"
        "        ROUND(" + score_expr + ", " + str(decimals) + ") AS raw_score\n"
        "    FROM base\n"
        "),\n"
        "bonus_calc AS (\n"
        "    SELECT\n"
        "        " + pk + ",\n"
        "        is_veto,\n"
        "        bonus_points,\n"
        "        bonus_pct,\n"
        "        CASE WHEN raw_score < 0 THEN 0.000000 ELSE raw_score END AS base_pos\n"
        "    FROM scored\n"
        "),\n"
        "final AS (\n"
        "    SELECT\n"
        "        " + pk + ",\n"
        "        is_veto,\n"
        "        ROUND(\n"
        "            base_pos + LEAST(\n"
        "                bonus_points + ROUND((base_pos + bonus_points) * bonus_pct, " + str(decimals) + "),\n"
        "                " + fmt_sql_number(max_total) + "\n"
        "            ),\n"
        "            " + str(decimals) + "\n"
        "        ) AS " + score_alias + "\n"
        "    FROM bonus_calc\n"
        ")\n"
        "SELECT\n"
        "    " + pk + ",\n"
        "    " + score_alias + ",\n"
        "    CASE\n"
        "        WHEN is_veto = 1 THEN 'E'\n"
        "        WHEN " + score_alias + " >= " + format(t_a, ".2f") + " THEN 'A'\n"
        "        WHEN " + score_alias + " >= " + format(t_b, ".2f") + " THEN 'B'\n"
        "        WHEN " + score_alias + " >= " + format(t_c, ".2f") + " THEN 'C'\n"
        "        ELSE 'D'\n"
        "    END AS " + level_alias + "\n"
        "FROM final;\n"
    )
    return sql_text
