# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Characterization tests for the JUnit converter: they pin the decisions that make a
# row honest, so a later refactor cannot quietly change one. Each test names the lie
# it prevents rather than the branch it covers.
#
# Run: python3 -m unittest discover -s bin/rocke/tests
#
# The converter is exercised as a subprocess through its real CLI, because that is
# the only interface run_rocke.sh uses and an import-level test would not notice an
# argument that stopped being passed.

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONVERTER = HERE.parent / "rocke_junit_results.py"


def convert(xml: str, *args: str) -> list[list[str]]:
    """Run the converter over one JUnit document; return rows as split fields."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.xml"
        path.write_text(xml, encoding="utf-8")
        out = subprocess.run(
            [sys.executable, str(CONVERTER), "--junit", str(path),
             "--group-default", "pytest", *args],
            capture_output=True, text=True, check=True,
        ).stdout
    # Split on newlines only: str.splitlines() also breaks on \x1e and \x1f, which
    # are exactly the sentinels rocke_result.py encodes pipes and newlines with, so
    # it would tear a row apart and make the encoding look broken.
    return [line.split("|") for line in out.split("\n")
            if line.startswith("ROCKE_RESULT|")]


def suite(cases: str, **tally: int) -> str:
    attrs = " ".join(f'{k}="{v}"' for k, v in tally.items())
    return f'<testsuite name="pytest" {attrs}>{cases}</testsuite>'


class BlockedSkips(unittest.TestCase):
    """A skip is red only when the toolchain could not do the work."""

    def test_a_toolchain_that_cannot_compile_is_red_in_the_compiler_tier(self):
        rows = convert(suite(
            '<testcase classname="m.C" name="t">'
            '<skipped message="comgr (6, 4) cannot target gfx1250 (needs &gt;= 7.2)"/>'
            "</testcase>", tests=1, failures=0, skipped=1))
        self.assertEqual(rows[0][3], "1")
        self.assertEqual(rows[0][5], "compiler")

    def test_absent_upstream_golden_is_not_a_compiler_failure(self):
        # The reason names llvm only to say upstream has not recorded data yet.
        # Read as blocked, it is red in the compiler tier every night forever.
        rows = convert(suite(
            '<testcase classname="m.C" name="t">'
            "<skipped message=\"no gfx942 golden recorded for llvm flavor 'llvm23'\"/>"
            "</testcase>", tests=1, failures=0, skipped=1))
        self.assertEqual(rows[0][3], "Check")
        self.assertEqual(rows[0][5], "unmeasured")

    def test_an_environmental_skip_is_neither_pass_nor_failure(self):
        rows = convert(suite(
            '<testcase classname="m.C" name="t">'
            '<skipped message="needs a gfx950 GPU with ROCm torch"/>'
            "</testcase>", tests=1, failures=0, skipped=1))
        self.assertEqual(rows[0][3], "Check")


class DatalayoutDrift(unittest.TestCase):
    """Drift measured against another toolchain is not a verdict on the COD."""

    DRIFT = ("AssertionError: 'e-p:64:64' != 'e-m:e-p:64:64' : "
             "Datalayout drift detected for gfx11-generic under llvm20.")

    def _row(self, *args: str) -> list[str]:
        return convert(suite(
            f'<testcase classname="m.C" name="t"><failure message="{self.DRIFT}"/>'
            "</testcase>", tests=1, failures=1, skipped=0), *args)[0]

    def test_drift_under_another_flavor_is_unmeasured(self):
        row = self._row("--cod-flavor", "llvm23")
        self.assertEqual(row[3], "Check")
        self.assertIn("llvm20", row[4])
        self.assertIn("llvm23", row[4])

    def test_drift_under_the_cods_own_flavor_stays_red(self):
        self.assertEqual(self._row("--cod-flavor", "llvm20")[3], "1")

    def test_without_a_pinned_flavor_it_stays_red(self):
        self.assertEqual(self._row()[3], "1")


class RunnerTallies(unittest.TestCase):
    """The runner counts what it ran; we count what it wrote down."""

    def test_subtest_collapse_is_accepted(self):
        # pytest-subtests counts each failing subtest in the header while the body
        # carries one testcase per test. Reporting that as loss would redden a
        # correct report: 65 declared against 57 testcases is the normal case.
        rows = convert(suite(
            '<testcase classname="m.C" name="t"><failure message="boom"/></testcase>',
            tests=4, failures=3, skipped=0))
        self.assertFalse([r for r in rows if "junit-tally" in r[2]])

    def test_a_declared_failure_that_reaches_no_row_is_reported(self):
        rows = convert(suite(
            '<testcase classname="m.C" name="t"/>', tests=1, failures=2, skipped=0))
        tally = [r for r in rows if "junit-tally" in r[2]]
        self.assertEqual(len(tally), 1)
        self.assertEqual(tally[0][3], "1")
        self.assertEqual(tally[0][5], "harness")

    def test_a_skip_that_did_not_become_a_row_is_reported(self):
        rows = convert(suite(
            '<testcase classname="m.C" name="t"/>', tests=1, failures=0, skipped=3))
        self.assertTrue([r for r in rows if "junit-tally" in r[2]])

    def test_a_runner_without_tallies_is_not_second_guessed(self):
        rows = convert('<testsuite name="ctest">'
                       '<testcase classname="m.C" name="t"/></testsuite>')
        self.assertFalse([r for r in rows if "junit-tally" in r[2]])


class Coverage(unittest.TestCase):
    """Losing rows or losing relevance must never be silent."""

    def test_an_empty_report_is_red(self):
        rows = convert('<testsuite name="pytest"></testsuite>')
        self.assertEqual(rows[0][2], "pytest-no-testcases")
        self.assertEqual(rows[0][3], "1")

    def test_a_module_skipped_at_collection_is_not_a_lost_measurement(self):
        # No test item ever existed, so no relevance entry can exist for it.
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "relevance.json"
            manifest.write_text('{"version": 0, "tests": {}}', encoding="utf-8")
            rows = convert(suite(
                '<testcase classname="" name="m.mod">'
                '<skipped message="collection skipped"/></testcase>',
                tests=1, failures=0, skipped=1),
                "--relevance", str(manifest), "--relevance-default", "unmeasured")
        self.assertFalse([r for r in rows if "relevance-join" in r[2]])

    def test_an_unreadable_manifest_is_reported(self):
        rows = convert(suite('<testcase classname="m.C" name="t"/>',
                             tests=1, failures=0, skipped=0),
                       "--relevance", "/nonexistent/relevance.json")
        self.assertTrue([r for r in rows if "relevance-probe" in r[2] and r[3] == "1"])


class RowShape(unittest.TestCase):
    """Every row carries six fields, whatever the message contains."""

    def test_pipes_in_a_message_cannot_add_fields(self):
        rows = convert(suite(
            '<testcase classname="m.C" name="t">'
            '<failure message="a | b | c"/></testcase>',
            tests=1, failures=1, skipped=0))
        self.assertEqual(len(rows[0]), 6)
        self.assertEqual(rows[0][5], "unmeasured")


if __name__ == "__main__":
    unittest.main()
