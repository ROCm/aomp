# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The relevance vocabulary: how a red row in the rocKE nightly should be triaged.
#
# Its own module because it is shared by parts that must not depend on each other --
# the pytest plugin that measures relevance, the JUnit converter, the toolchain interop
# probes, and the CI-side extractor in the apps repo. Folded into the plugin, every
# one of those had to import a pytest plugin to read five strings.

TIER_COMPILER = "compiler"  # this test called the toolchain itself
TIER_CAPABLE = "compiler-capable"  # it did not, but its module does, or it spawned a child
TIER_LOGIC = "logic"  # no toolchain interaction anywhere in the module
TIER_HARNESS = "harness"  # this CI's own plumbing
TIER_UNMEASURED = "unmeasured"  # lane carries no per-test evidence

# Most to least compiler-relevant, so a consumer can pick the stronger of two
# verdicts (see the converter's relevance floor) and order a report by triage value.
TIER_ORDER = (TIER_COMPILER, TIER_CAPABLE, TIER_UNMEASURED, TIER_HARNESS, TIER_LOGIC)
