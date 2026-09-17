# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Convert a JUnit XML report (pytest --junitxml or ctest --output-junit) into the
# canonical rocKE CI result lines consumed by extract-rocke.sh:
#
#   ROCKE_RESULT|<group>|<subtest>|<status>|<message>|<relevance>
#
# status is 0 (pass), 1 (fail), or Check (nothing was measurable). A skip is Check
# only when the *host* is what is missing -- no GPU of the right arch, no ROCm torch
# -- so a GPU-free host is not reported as failing, and equally not as passing. A skip
# whose reason names the toolchain is a red "blocked: ..." row instead: it means the compiler under test could not do the
# work, which is the one thing this CI exists to catch. --relevance attaches the
# per-test evidence rocke_relevance.py recorded; --relevance-default covers lanes
# with none.

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET

try:
    from rocke_relevance import MANIFEST_VERSION
except ModuleNotFoundError:
    # This converter is useful without the pytest plugin: a lane whose runner writes
    # JUnit but measures nothing per test (ctest) needs the rows, not the manifest.
    # Only manifest handling needs the plugin's format version, and a manifest cannot
    # exist unless the plugin wrote one -- so if one is passed anyway, None makes the
    # version check reject it, which is the honest verdict rather than a crash.
    MANIFEST_VERSION = None

from rocke_tiers import TIER_COMPILER, TIER_HARNESS, TIER_ORDER, TIER_UNMEASURED
from rocke_result import STATUS_CHECK
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

# A skip whose reason names the toolchain only because it reports *absent upstream
# data*: no golden was recorded for a flavor, so nothing was ever asked of the
# compiler. The blocked rule above matches such a reason on the bare word "llvm" and
# turns "upstream has not got here yet" into a compiler failure -- red every night in
# the one tier triage is told to read first.
#
# Anchored to the whole reason, not searched within it. A reason that says both
# things -- "no golden recorded; comgr cannot target gfx950" -- describes a compiler
# that could not do the work, and a suppression that wins over a blocker is the one
# mistake this rule must never make. Anything reworded or extended upstream stops
# matching and reddens, which is the safe direction.
_MISSING_UPSTREAM_DATA = re.compile(
    r"\s*no (?:\S+ )*golden recorded for llvm flavor '[^']*'\s*\Z",
    re.IGNORECASE,
)

# rocKE's datalayout drift guard validates the constant for the flavor it reads from
# the *host* /opt/rocm against IR emitted by the hipcc on PATH -- which this CI points
# at the COD. On a host whose ROCm is a different vintage than the COD, the two halves
# come from different toolchains and the comparison says nothing about the COD, so it
# would sit red every night in the tier that must stay worth reading. The failure names
# the flavor it used, which is the evidence: reported under any flavor but the COD's,
# nothing about the COD was measured. Drift under the COD's own flavor stays red --
# that is the case the guard exists for. run_rocke.sh's own cod-datalayout probe checks
# the COD end from the clang's emitted shape, which no host ROCm can reach.
_DATALAYOUT_DRIFT = re.compile(r"[Dd]atalayout drift detected for \S+ under (\S+?)\b")


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


def _declared(root, name: str) -> int | None:  # noqa: ANN001
    """Sum of a <testsuite> tally attribute, or None when the runner omits it."""
    found = [s.get(name) for s in root.iter("testsuite") if s.get(name) is not None]
    try:
        return sum(int(v) for v in found) if found else None
    except ValueError:
        return None


def _reconcile(group_default: str, root, counts: dict[str, int]) -> None:  # noqa: ANN001
    """Check what the report carries against the tallies the runner declared.

    Counted in the unit the runner counts in: outcome *elements*, not testcases.
    pytest-subtests writes one testcase per test but a <failure> per failing subtest,
    so a report declaring 65 failures carries 65 <failure> elements across 57
    testcases. Comparing against testcases would have meant tolerating any number
    between 1 and 65, which is no check at all: 64 failures could vanish unnoticed.

    A runner that reports an outcome through a status attribute instead of an element
    is counted too, so ctest -- which writes `status` and declares `failures` -- is
    measured the same way.

    Each attribute is compared only where the runner declares it, because an absent
    attribute is not a claim of zero, and a false red here would be its own lie.
    """
    for attr, seen in counts.items():
        declared = _declared(root, attr)
        if declared is not None and declared != seen:
            _emit(
                "setup",
                f"{group_default}-junit-tally",
                1,
                f"runner declared {declared} {attr}, the report carries {seen}",
                TIER_HARNESS,
            )


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
    ap.add_argument(
        "--cod-flavor",
        default="",
        help="IR flavor the COD speaks, so a check that measured another "
        "toolchain's flavor is reported as unmeasured rather than as a COD failure",
    )
    ap.add_argument(
        "--cod-datalayout",
        default="",
        help="datalayout the COD emits, carried on such a row so it still records "
        "what this compiler did",
    )
    args = ap.parse_args()

    tiers: dict[str, str] = {}
    problems: list[str] = []
    if args.relevance:
        tiers, problems = _load_manifest(args.relevance)
    for problem in problems:
        _emit("setup", f"{args.group_default}-relevance-probe", 1, problem, TIER_HARNESS)

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
    # Outcome elements, in the runner's own unit -- see _reconcile.
    counts = {"failures": 0, "errors": 0, "skipped": 0}
    for case in root.iter("testcase"):
        seen += 1
        for attr, tag in (("failures", "failure"), ("errors", "error"),
                          ("skipped", "skipped")):
            counts[attr] += len(case.findall(tag))
        group = case.get("classname") or args.group_default
        subtest = case.get("name") or "unnamed"
        tier = tiers.get(f"{case.get('classname') or ''}\t{case.get('name') or ''}")
        if tier is None:
            tier = args.relevance_default
            # A module skipped at collection never became a test item, so the plugin
            # had nothing to record and the manifest cannot hold an entry. pytest
            # reports these with no classname. Counting them as lost relevance turned
            # "this host has no gfx1250" into a red harness row every night.
            if case.get("classname"):
                unjoined += 1
        tier = _at_least(tier, args.relevance_floor)
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        status_attr = (case.get("status") or "").lower()

        # A runner that reports through `status` alone still counts, so the tally
        # below measures ctest's shape as well as pytest's.
        if failure is None and error is None and status_attr in ("fail", "failed"):
            counts["failures"] += 1
        if skipped is None and status_attr in ("notrun", "disabled", "skipped"):
            counts["skipped"] += 1

        if (
            failure is not None
            or error is not None
            or status_attr in ("fail", "failed")
        ):
            node = failure if failure is not None else error
            msg = (node.get("message") if node is not None else "") or "failed"
            drift = args.cod_flavor and _DATALAYOUT_DRIFT.search(msg)
            # The COD's own datalayout goes on the row: the check is not a verdict,
            # but what this compiler emits is still worth recording where the
            # suppression happens.
            codLayout = (
                f". The COD emits: {args.cod_datalayout}" if args.cod_datalayout else ""
            )
            if drift and drift.group(1) != args.cod_flavor:
                _emit(
                    group,
                    subtest,
                    STATUS_CHECK,
                    f"measured the host toolchain's {drift.group(1)} datalayout, "
                    f"not the COD's {args.cod_flavor}, so it is no verdict on this "
                    f"compiler{codLayout}",
                    TIER_UNMEASURED,
                )
            else:
                _emit(group, subtest, 1, msg, tier)
        elif skipped is not None or status_attr in ("notrun", "disabled", "skipped"):
            reason = (
                skipped.get("message") if skipped is not None else ""
            ) or "skipped"
            if _BLOCKED_SKIP.search(reason) and not _MISSING_UPSTREAM_DATA.match(
                reason
            ):
                blocked_tier = (
                    TIER_HARNESS if _BLOCKED_OURS.search(reason) else TIER_COMPILER
                )
                _emit(group, subtest, 1, f"blocked: {reason}", blocked_tier)
            else:
                # A test that never ran recorded no evidence, so the manifest tier
                # is spurious -- and lands on 'logic', which reads as "not the
                # compiler's business". Say outright that nothing was measured.
                _emit(group, subtest, STATUS_CHECK, reason, TIER_UNMEASURED)
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
    # instead of reporting everything as unmeasured. Guarding this on a non-empty
    # manifest would have made the *partial* loss loud and the total loss silent,
    # since a manifest that joined nothing at all leaves `tiers` empty.
    elif args.relevance and unjoined:
        _emit(
            "setup",
            f"{args.group_default}-relevance-join",
            1,
            f"{unjoined} of {seen} testcases had no relevance entry"
            f"{' (the manifest matched nothing)' if not tiers else ''}",
            TIER_HARNESS,
        )

    if seen:
        _reconcile(args.group_default, root, counts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
