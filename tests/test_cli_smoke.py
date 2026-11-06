# tests/test_cli_smoke.py
"""统一入口冒烟：list / init / list，并清理临时场景。"""
import os
import shutil
import subprocess
import sys
import time
import unittest
import uuid

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIO_DIR = os.path.join(ROOT_DIR, "scenarios")


def _run(*args):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT_DIR, "run.py")] + list(args),
        cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
    )


def _rmtree(path, attempts=5):
    """Windows 下目录可能被索引/杀毒短暂占用，重试删除"""
    for _ in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(0.2)
    shutil.rmtree(path)


class CliSmokeTest(unittest.TestCase):
    def test_list_contains_example(self):
        result = _run("list")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("industry_finance", result.stdout)

    def test_init_and_list_temporary_scenario(self):
        name = "smoke_" + uuid.uuid4().hex[:8]
        target = os.path.join(SCENARIO_DIR, name)
        try:
            result = _run("init", "--scenario", name)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.exists(os.path.join(target, "scenario.yaml")))

            listed = _run("list")
            self.assertEqual(listed.returncode, 0, listed.stderr)
            self.assertIn(name, listed.stdout)
        finally:
            if os.path.isdir(target):
                _rmtree(target)


if __name__ == "__main__":
    unittest.main()
