# tests/test_rules.py
"""业务规则：否决缺失语义、加分 all/any 组合、规则字段收集与配置校验。"""
import os
import sys
import unittest

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.rules import (  # noqa: E402
    rule_fields,
    rule_mask,
    validate_rules,
    veto_mask,
)


def _cfg(veto_conditions=None, bonus=None, fields=None):
    fields = fields or [{"name": "age", "cn_name": "年龄", "data_type": "int", "feature_type": "numeric"}]
    rules = {}
    if veto_conditions is not None:
        rules["veto"] = {"enabled": True, "exclude_from_train": True, "any": veto_conditions}
    if bonus is not None:
        rules["bonus"] = bonus
    cfg = {"fields": fields, "_field_map": {f["name"]: f for f in fields}, "rules": rules}
    return cfg


class VetoMaskTest(unittest.TestCase):
    def test_string_equals(self):
        cfg = _cfg([{"field": "reg_st", "op": "=", "value": "正常"}])
        df = pd.DataFrame({"reg_st": ["正常", "异常", None]})
        self.assertEqual(veto_mask(df, cfg).tolist(), [True, False, False])

    def test_string_not_equal_hits_missing(self):
        cfg = _cfg([{"field": "reg_st", "op": "!=", "value": "正常"}])
        df = pd.DataFrame({"reg_st": ["正常", "异常", None]})
        self.assertEqual(veto_mask(df, cfg).tolist(), [False, True, True])

    def test_numeric_greater_than_ignores_missing(self):
        cfg = _cfg([{"field": "age", "op": ">", "value": 5}])
        df = pd.DataFrame({"age": [6, None, 3]})
        self.assertEqual(veto_mask(df, cfg).tolist(), [True, False, False])

    def test_any_condition_union(self):
        cfg = _cfg([
            {"field": "a", "op": "=", "value": "1"},
            {"field": "b", "op": "=", "value": "1"},
        ])
        df = pd.DataFrame({"a": ["1", "0", "0", None], "b": ["0", "1", "0", "1"]})
        self.assertEqual(veto_mask(df, cfg).tolist(), [True, True, False, True])


class BonusRuleTest(unittest.TestCase):
    def test_all_and_any_combination(self):
        rule = {
            "all": [{"field": "flag_a", "op": "=", "value": "1"}],
            "any": [
                {"field": "entp_scl", "op": "=", "value": "中型"},
                {"field": "entp_scl", "op": "=", "value": "大型"},
            ],
            "points": 5,
        }
        df = pd.DataFrame({
            "flag_a": ["1", "1", "0", "1"],
            "entp_scl": ["中型", "小型", "大型", None],
        })
        self.assertEqual(rule_mask(df, rule).tolist(), [True, False, False, False])

    def test_rule_fields_dedupes(self):
        bonus = {
            "enabled": True,
            "score_rules": [
                {"all": [{"field": "a", "op": "=", "value": "1"}], "points": 5},
                {"any": [{"field": "b", "op": ">", "value": 1}], "points": 5},
            ],
            "percent_rules": [],
        }
        cfg = _cfg(
            [{"field": "reg_st", "op": "!=", "value": "正常"}, {"field": "a", "op": "=", "value": "1"}],
            bonus,
        )
        self.assertEqual(rule_fields(cfg), ["reg_st", "a", "b"])


class ValidateRulesTest(unittest.TestCase):
    def test_veto_field_cannot_be_feature(self):
        cfg = _cfg([{"field": "age", "op": ">", "value": 5}])
        with self.assertRaises(ValueError):
            validate_rules(cfg, "scenario.yaml")

    def test_string_condition_requires_equality_operator(self):
        cfg = _cfg([{"field": "reg_st", "op": ">", "value": "正常"}])
        with self.assertRaises(ValueError):
            validate_rules(cfg, "scenario.yaml")

    def test_bonus_requires_condition(self):
        cfg = _cfg(None, {"enabled": True, "score_rules": [{"points": 5}], "percent_rules": []})
        with self.assertRaises(ValueError):
            validate_rules(cfg, "scenario.yaml")


if __name__ == "__main__":
    unittest.main()
