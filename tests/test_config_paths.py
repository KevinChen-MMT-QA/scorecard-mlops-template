# tests/test_config_paths.py
"""配置加载：相对路径按场景目录解析、字段清单与主键校验。"""
import os
import sys
import tempfile
import textwrap
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.config import load_config  # noqa: E402

FIELDS_CSV = textwrap.dedent("""\
    name,cn_name,data_type,feature_type
    age,年龄,int,numeric
    flag_a,标志位,string,flag
    """)


def _write_scenario(base_dir, primary_key="unt_scl_cdt_cd"):
    with open(os.path.join(base_dir, "fields.csv"), "w", encoding="utf-8") as f:
        f.write(FIELDS_CSV)
    content = textwrap.dedent("""\
        scenario: demo
        fields_file: fields.csv
        source:
          database: db
          table_name: tbl
          primary_key: {pk}
          partition_field: part_dt
        data:
          load_mode: mock
          sample_format: csv
          pos_sample_csv: samples/pos.csv
          neg_sample_csv: samples/neg.csv
        sql:
          output_file: report/card.sql
          report_dir: report
        """).format(pk=primary_key)
    path = os.path.join(base_dir, "scenario.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class ConfigPathTest(unittest.TestCase):
    def test_relative_paths_resolve_to_scenario_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_scenario(tmp)
            cfg = load_config(config_path)
            self.assertEqual(cfg["_scenario_dir"], os.path.abspath(tmp))
            self.assertEqual(os.path.normpath(cfg["data"]["pos_sample_csv"]),
                             os.path.normpath(os.path.join(tmp, "samples", "pos.csv")))
            self.assertEqual(os.path.normpath(cfg["data"]["neg_sample_csv"]),
                             os.path.normpath(os.path.join(tmp, "samples", "neg.csv")))
            self.assertEqual(os.path.normpath(cfg["sql"]["output_file"]),
                             os.path.normpath(os.path.join(tmp, "report", "card.sql")))
            self.assertEqual(cfg["sql"]["report_dir"], os.path.join(tmp, "report"))
            self.assertEqual([f["name"] for f in cfg["fields"]], ["age", "flag_a"])

    def test_wrong_primary_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_scenario(tmp, primary_key="other_id")
            with self.assertRaises(ValueError):
                load_config(config_path)

    def test_missing_fields_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scenario.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("scenario: demo\nsource:\n  primary_key: unt_scl_cdt_cd\n")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
