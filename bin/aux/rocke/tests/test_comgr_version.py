# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Tests for the policy that decides which ROCm version rocKE sees. Getting this
# wrong is costly in both directions: leaving a foreign version in place makes rocKE
# refuse architectures the compiler supports, and overriding a version that can be
# believed would also satisfy rocKE's IR-flavor guard and cover up a real mismatch.
#
# Run: python3 -m unittest discover -s bin/aux/rocke/tests

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rocke_comgr_version as version_policy  # noqa: E402


class Comgr:
    """Stands in for rocKE's comgr module, reporting the system ROCm."""

    __name__ = "comgr"

    @staticmethod
    def resolved_lib_rocm_version() -> tuple[int, int]:
        return (6, 4)


class Policy(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {k: os.environ.get(k) for k in
                      (version_policy.TRUSTED_ENV, version_policy.BUILD_VERSION_ENV)}
        self.comgr = Comgr()

    def tearDown(self) -> None:
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def env(self, trusted: str, build: str = "") -> None:
        os.environ[version_policy.TRUSTED_ENV] = trusted
        os.environ[version_policy.BUILD_VERSION_ENV] = build

    def test_a_version_that_can_be_believed_is_left_alone(self):
        self.env("1", "10.2")
        self.assertIsNone(version_policy.pinWhenUntrusted(self.comgr))
        self.assertEqual(self.comgr.resolved_lib_rocm_version(), (6, 4))

    def test_the_builds_version_replaces_a_foreign_one(self):
        self.env("0", "10.2")
        self.assertEqual(version_policy.pinWhenUntrusted(self.comgr), (10, 2))
        self.assertEqual(self.comgr.resolved_lib_rocm_version(), (10, 2))

    def test_no_build_version_and_no_flavor_changes_nothing(self):
        self.env("0")
        self.assertIsNone(version_policy.pinWhenUntrusted(self.comgr))
        self.assertEqual(self.comgr.resolved_lib_rocm_version(), (6, 4))

    def test_an_unset_trust_flag_changes_nothing(self):
        os.environ.pop(version_policy.TRUSTED_ENV, None)
        os.environ[version_policy.BUILD_VERSION_ENV] = "10.2"
        self.assertIsNone(version_policy.pinWhenUntrusted(self.comgr))

    def test_a_malformed_build_version_is_ignored(self):
        self.env("0", "not-a-version")
        self.assertIsNone(version_policy.buildVersion())

    def test_a_renamed_upstream_symbol_is_reported(self):
        class Moved:
            __name__ = "comgr"

        with self.assertRaises(AttributeError):
            version_policy.pin(Moved(), (10, 2))


if __name__ == "__main__":
    unittest.main()
