#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Convert a JUnit XML report (pytest --junitxml or ctest --output-junit) into the
# canonical rocKE CI result lines consumed by extract-rocke.sh:
#
#   ROCKE_RESULT|<group>|<subtest>|<status>|<message>|<relevance>
#
# status is 0 (pass) or 1 (fail). A skip is a pass ("skipped: ...") only when the
# *host* is what is missing -- no GPU of the right arch, no ROCm torch -- so a
# GPU-free host stays green. A skip whose reason names the toolchain is a red
# "blocked: ..." row instead: it means the compiler under test could not do the
# work, which is the one thing this CI exists to catch. --relevance attaches the
# per-test evidence rocke_relevance.py recorded; --relevance-default covers lanes
# with none.

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET

from rocke_relevance import (
    MANIFEST_VERSION,
    TIER_COMPILER,
    TIER_HARNESS,
    TIER_ORDER,
    TIER_UNMEASURED,
)
from rocke_result import emit as _emit

# Toolchain markers in a skip reason. Kept deliberately narrow and extended only
# from reasons actually observed, so an unrecognised reason stays green rather than
# turning the nightly red on a guess.
_BLOCKED_SKIP = re.compile(
    r"comgr|hipcc|\bclang\b|llvm|\bisa\b|cannot target|compile unavailable"
    r"|rocke_engine|c\+\+ engine|byte.identity|datalayout",
    re.IGNORECASE,
)

# Of those, the ones naming an artifact *this CI* is responsible for building. Still
# red -- the tests did not run -- but ours to fix, so pointing triage at the compiler
# would waste its time. A COD failure while building it is reported by the builder.
_BLOCKED_OURS = re.compile(r"rocke_engine|c\+\+ engine", re.IGNORECASE)


def _at_least(tier: str, floor: str) -> str:
    """The more compiler-relevant of the two, when a floor is given.

    Some lanes drive the toolchain by construction -- every case in the numeric lane
    compiles a kernel and launches it -- yet the measured evidence can be empty
    because the work happened in a child process. Reporting such a row as `logic`
    ("not the compiler's business") is worse than reporting no measurement at all,
    so the lane's guarantee wins.
    """
    if not floor or floor not in TIER_ORDER or tier not in TIER_ORDER:
        return tier
    return min(tier, floor, key=TIER_ORDER.index)


def _load_manifest(path: str) -> tuple[dict[str, str], list[str]]:
    """(key -> tier, setup problems worth a red row).

    An entry with no usable tier is dropped rather than kept as "", so it counts
    as unjoined below instead of silently degrading the signal.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        return {}, [f"cannot read relevance manifest {path}: {exc}"]
    # The plugin stamps the layout it wrote; refuse to read tiers out of a shape
    # this decoder does not know rather than mislabel every row from it.
    found = data.get("version")
    if found != MANIFEST_VERSION:
        return {}, [
            f"relevance manifest {path} is version {found!r}, expected {MANIFEST_VERSION}"
        ]
    tests = (data.get("tests") or {}).items()
    tiers = {k: t for k, v in tests if (t := str(v.get("tier") or ""))}
    return tiers, list(data.get("install_errors") or [])


def main() -> int:
    ap = argparse.ArgumentParser(description="JUnit XML -> rocKE CI result lines")
    ap.add_argument("--junit", required=True, help="path to the JUnit XML report")
    ap.add_argument(
        "--group-default",
        default="tests",
        help="group used when a testcase has no classname",
    )
    ap.add_argument("--relevance", help="path to the rocke_relevance.py manifest")
    ap.add_argument(
        "--relevance-default",
        default=TIER_UNMEASURED,
        help="relevance for cases the manifest does not cover",
    )
    ap.add_argument(
        "--relevance-floor",
        default="",
        help="least relevance a case that ran may report, for a lane whose every "
        "test drives the toolchain by construction (e.g. on-device numerics)",
    )
    args = ap.parse_args()

    tiers: dict[str, str] = {}
    problems: list[str] = []
    if args.relevance:
        tiers, problems = _load_manifest(args.relevance)
    for problem in problems:
        _emit("setup", "relevance-probe", 1, problem, TIER_HARNESS)

    try:
        root = ET.parse(args.junit).getroot()
    except (OSError, ET.ParseError) as exc:
        _emit(
            "setup",
            f"{args.group_default}-junit-parse",
            1,
            f"cannot parse {args.junit}: {exc}",
            TIER_HARNESS,
        )
        return 0

    seen = 0
    unjoined = 0
    for case in root.iter("testcase"):
        seen += 1
        group = case.get("classname") or args.group_default
        subtest = case.get("name") or "unnamed"
        tier = tiers.get(f"{case.get('classname') or ''}\t{case.get('name') or ''}")
        if tier is None:
            tier = args.relevance_default
            unjoined += 1
        tier = _at_least(tier, args.relevance_floor)
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        status_attr = (case.get("status") or "").lower()

        if (
            failure is not None
            or error is not None
            or status_attr in ("fail", "failed")
        ):
            node = failure if failure is not None else error
            msg = (node.get("message") if node is not None else "") or "failed"
            _emit(group, subtest, 1, msg, tier)
        elif skipped is not None or status_attr in ("notrun", "disabled", "skipped"):
            reason = (
                skipped.get("message") if skipped is not None else ""
            ) or "skipped"
            if _BLOCKED_SKIP.search(reason):
                blocked_tier = (
                    TIER_HARNESS if _BLOCKED_OURS.search(reason) else TIER_COMPILER
                )
                _emit(group, subtest, 1, f"blocked: {reason}", blocked_tier)
            else:
                # A test that never ran recorded no evidence, so the manifest tier
                # is spurious -- and lands on 'logic', which reads as "not the
                # compiler's business". Say outright that nothing was measured.
                _emit(group, subtest, 0, f"skipped: {reason}", TIER_UNMEASURED)
        else:
            _emit(group, subtest, 0, "", tier)

    if seen == 0:
        _emit(
            "setup",
            f"{args.group_default}-no-testcases",
            1,
            "report contained no testcases",
            TIER_HARNESS,
        )
    # A manifest that stops joining silently loses the compiler signal, so say so
    # instead of reporting everything as unmeasured.
    elif tiers and unjoined:
        _emit(
            "setup",
            "relevance-join",
            1,
            f"{unjoined} of {seen} testcases had no relevance entry",
            TIER_HARNESS,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
