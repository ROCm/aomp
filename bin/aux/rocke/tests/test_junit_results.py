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

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONVERTER = HERE.parent / "rocke_junit_results.py"
sys.path.insert(0, str(HERE.parent))

from rocke_relevance import MANIFEST_VERSION, claim_device_for_torch  # noqa: E402


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

    def test_a_reason_saying_both_things_stays_red(self):
        # A suppression that wins over a genuine blocker is the one mistake this
        # rule must never make, so the exemption matches the whole reason or not
        # at all.
        rows = convert(suite(
            '<testcase classname="m.C" name="t">'
            "<skipped message=\"no gfx942 golden recorded for llvm flavor 'llvm23'"
            '; comgr cannot target gfx950"/>'
            "</testcase>", tests=1, failures=0, skipped=1))
        self.assertEqual(rows[0][3], "1")
        self.assertEqual(rows[0][5], "compiler")

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

    def tally(self, rows: list[list[str]]) -> list[list[str]]:
        return [r for r in rows if "junit-tally" in r[2]]

    def test_subtests_are_counted_in_the_runners_unit(self):
        # pytest-subtests writes one testcase per test but a <failure> per failing
        # subtest, so a report declaring 3 failures carries 3 elements in 1
        # testcase. Counting testcases instead would tolerate any number from 1 to
        # 3, which is no check at all.
        rows = convert(suite(
            '<testcase classname="m.C" name="t">'
            '<failure message="a"/><failure message="b"/><failure message="c"/>'
            "</testcase>", tests=4, failures=3, skipped=0))
        self.assertFalse(self.tally(rows))

    def test_losing_all_but_one_failure_is_reported(self):
        rows = convert(suite(
            '<testcase classname="m.C" name="t"><failure message="a"/></testcase>',
            tests=4, failures=3, skipped=0))
        self.assertEqual(len(self.tally(rows)), 1)
        self.assertEqual(self.tally(rows)[0][3], "1")
        self.assertEqual(self.tally(rows)[0][5], "harness")

    def test_a_declared_failure_that_reaches_no_row_is_reported(self):
        rows = convert(suite(
            '<testcase classname="m.C" name="t"/>', tests=1, failures=2, skipped=0))
        self.assertEqual(len(self.tally(rows)), 1)

    def test_a_skip_that_did_not_become_a_row_is_reported(self):
        rows = convert(suite(
            '<testcase classname="m.C" name="t"/>', tests=1, failures=0, skipped=3))
        self.assertTrue(self.tally(rows))

    def test_a_runner_without_tallies_is_not_second_guessed(self):
        rows = convert('<testsuite name="ctest">'
                       '<testcase classname="m.C" name="t"/></testsuite>')
        self.assertFalse(self.tally(rows))

    def test_an_outcome_reported_only_by_status_counts(self):
        # ctest declares failures and marks the case with an attribute rather than
        # an element; counting elements alone would report a false loss.
        rows = convert('<testsuite name="ctest" tests="1" failures="1">'
                       '<testcase classname="m.C" name="t" status="fail"/>'
                       "</testsuite>")
        self.assertFalse(self.tally(rows))

    def test_an_attribute_the_runner_omits_is_not_read_as_zero(self):
        rows = convert('<testsuite name="ctest" tests="1" failures="1">'
                       '<testcase classname="m.C" name="t">'
                       '<failure message="boom"/></testcase></testsuite>')
        self.assertFalse(self.tally(rows))

    def test_tallies_across_several_suites_are_summed(self):
        rows = convert(
            '<testsuites><testsuite name="a" tests="1" failures="1">'
            '<testcase classname="m.A" name="t"><failure message="x"/></testcase>'
            '</testsuite><testsuite name="b" tests="1" failures="1">'
            '<testcase classname="m.B" name="t"><failure message="y"/></testcase>'
            "</testsuite></testsuites>")
        self.assertFalse(self.tally(rows))


class Coverage(unittest.TestCase):
    """Losing rows or losing relevance must never be silent."""

    def test_an_empty_report_is_red(self):
        rows = convert('<testsuite name="pytest"></testsuite>')
        self.assertEqual(rows[0][2], "pytest-no-testcases")
        self.assertEqual(rows[0][3], "1")

    def test_a_module_skipped_at_collection_is_not_a_lost_measurement(self):
        # No test item ever existed, so no relevance entry can exist for it. The
        # manifest has to carry the current version, or the converter rejects it
        # and the test would pass on the wrong row.
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "relevance.json"
            manifest.write_text(f'{{"version": {MANIFEST_VERSION}, "tests": {{}}}}',
                                encoding="utf-8")
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


class DeviceClaim(unittest.TestCase):
    """Claiming the context must never become a way to lose a result."""

    def setUp(self):
        self.saved = os.environ.get("ROCKE_CLAIM_DEVICE_FOR_TORCH")
        import rocke_relevance
        rocke_relevance._S.device_claimed = False

    def tearDown(self):
        os.environ.pop("ROCKE_CLAIM_DEVICE_FOR_TORCH", None)
        if self.saved is not None:
            os.environ["ROCKE_CLAIM_DEVICE_FOR_TORCH"] = self.saved

    def test_disabling_it_says_nothing_and_does_nothing(self):
        os.environ["ROCKE_CLAIM_DEVICE_FOR_TORCH"] = "0"
        self.assertEqual(claim_device_for_torch(), "")

    def test_a_host_without_torch_is_silent(self):
        # Absence of torch is reported by the lane that needs it; a line per run
        # about a dependency nobody asked for is noise.
        os.environ["ROCKE_CLAIM_DEVICE_FOR_TORCH"] = "1"
        if _torch_present():
            self.skipTest("torch is installed here")
        self.assertEqual(claim_device_for_torch(), "")

    def test_it_claims_once(self):
        os.environ["ROCKE_CLAIM_DEVICE_FOR_TORCH"] = "1"
        first = claim_device_for_torch()
        self.assertEqual(claim_device_for_torch(), "")
        if _torch_present():
            self.assertIn("torch", first)

    def test_a_torch_that_cannot_claim_reports_rather_than_raises(self):
        os.environ["ROCKE_CLAIM_DEVICE_FOR_TORCH"] = "1"
        import rocke_relevance
        broken = type(sys)("torch")
        broken.cuda = type("cuda", (), {"is_available": staticmethod(
            lambda: (_ for _ in ()).throw(RuntimeError("no driver")))})
        sys.modules["torch"] = broken
        try:
            rocke_relevance._S.device_claimed = False
            self.assertIn("could not claim", claim_device_for_torch())
        finally:
            del sys.modules["torch"]


def _torch_present() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None


if __name__ == "__main__":
    unittest.main()
