#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Per-test COD-toolchain relevance for the rocKE nightly, as a pytest plugin.
#
# Whether a red row can be a compiler regression is a property of what a test
# *did*, not of what it imports: rocke/__init__.py eagerly imports the comgr, HIP
# and hipcc entry points, so import-graph analysis marks nearly every test capable
# and decides nothing (see README.md "Test relevance"). Measure it instead, in
# decreasing precision:
#   1. rocke.runtime._ctypes_bind._LazyFn.__call__ -- the chokepoint every comgr
#      and HIP native call goes through; level-triggered, so it sees every call.
#   2. ctypes.dlopen / subprocess.Popen audit hooks -- a COD library load or a
#      hipcc/clang/llvm-* spawn from anywhere, including code this plugin knows
#      nothing about. Edge-triggered backstop.
#   3. the rocke_engine import -- the C++ engine is built by the COD.
#
# No allowlist, so a new test that compiles is flagged the moment it does. A
# chokepoint lost to a rocke refactor becomes an install error and then a red
# setup row, so staleness is loud rather than silent.

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

# Relevance of a result row, i.e. how a red row should be triaged.
TIER_COMPILER = "compiler"  # this test drove the COD toolchain
TIER_CAPABLE = "compiler-capable"  # its module did, this test did not get there
TIER_LOGIC = "logic"  # no toolchain interaction anywhere in the module
TIER_HARNESS = "harness"  # this CI's own plumbing
TIER_UNMEASURED = "unmeasured"  # lane carries no per-test evidence

# Most to least compiler-relevant, so a consumer can pick the stronger of two
# verdicts (see rocke_junit_results.py's floor) and order a report by triage value.
TIER_ORDER = (TIER_COMPILER, TIER_CAPABLE, TIER_UNMEASURED, TIER_HARNESS, TIER_LOGIC)

MANIFEST_VERSION = 1

# Shared-library roles worth attributing; the path is kept too, since rocke may
# prefer a torch-bundled libamd_comgr over the COD one and the report must say so.
_LIB_ROLES = (
    ("comgr", re.compile(r"libamd_comgr|amd_comgr\.dll")),
    ("hiprtc", re.compile(r"hiprtc")),
    ("hip", re.compile(r"libamdhip64|amdhip64\.dll")),
    ("hsa", re.compile(r"libhsa-runtime")),
)

# Toolchain binaries; matched on argv0's basename.
_TOOL_BIN = re.compile(
    r"^(hipcc|hipconfig|amdclang\+*|clang\+*|clang-\d+|opt|llc|lld|ld\.lld"
    r"|llvm-[a-z0-9-]+|roc-obj[a-z-]*|rocprof[a-z0-9]*|amdgpu-arch|offload-arch)$"
)

_SESSION = "<session>"

# Python interpreters, matched on argv0's basename: a test that hands its real work
# to a child process (rocKE's numeric and gfx950 smoke tests both do) does all its
# compiling there, where the in-process chokepoint cannot see any of it.
_PY_BIN = re.compile(r"^python[0-9.]*(\.exe)?$")


def _rocke_tree() -> str:
    """Root of the rocKE checkout, or "" when it cannot be located."""
    if _S.tree is None:
        _S.tree = ""
        try:
            import rocke

            # <tree>/platform/python/rocke/__init__.py -> <tree>
            _S.tree = os.path.realpath(
                os.path.join(os.path.dirname(rocke.__file__), "..", "..", "..")
            )
        except Exception:  # noqa: BLE001 - never break a spawn
            pass
    return _S.tree


def _rocke_child(argv) -> str:  # noqa: ANN001
    """What a spawned interpreter will run, when that code lives in the rocKE tree.

    Resolved against sys.path rather than matched by name, so a renamed package
    keeps being recognised -- exactly the staleness this design is meant to avoid.
    """
    tree = _rocke_tree()
    if not tree:
        return ""
    roots = [p for p in sys.path if p and os.path.realpath(p).startswith(tree)]
    for i, raw in enumerate(argv):
        arg = os.fsdecode(raw)
        if arg == "-m" and i + 1 < len(argv):
            target = os.fsdecode(argv[i + 1])
            top = target.split(".", 1)[0]
            if any(
                os.path.isdir(os.path.join(r, top))
                or os.path.exists(os.path.join(r, f"{top}.py"))
                for r in roots
            ):
                return target
        elif arg.endswith(".py") and os.path.realpath(arg).startswith(tree):
            return os.path.basename(arg)
    return ""


class _State:
    def __init__(self) -> None:
        self.current: list[str] = []
        self.evidence: dict[str, set[str]] = defaultdict(set)
        self.module_of: dict[str, str] = {}
        self.install_errors: list[str] = []
        self.tree: str | None = None


_S = _State()


def _record(kind: str) -> None:
    _S.evidence[_S.current[-1] if _S.current else _SESSION].add(kind)


def _audit(event, args):  # noqa: ANN001
    if event == "subprocess.Popen":
        exe = os.fsdecode(args[0]) if args[0] else ""
        argv = args[1] or []
        base = os.path.basename(exe or (os.fsdecode(argv[0]) if argv else ""))
        if _TOOL_BIN.match(base):
            _record(f"spawn:{base}")
        elif _PY_BIN.match(base):
            target = _rocke_child(argv)
            if target:
                _record(f"spawn:py:{target}")
    elif event == "ctypes.dlopen":
        name = str(args[0]) if args else ""
        for role, pat in _LIB_ROLES:
            if pat.search(name):
                _record(f"dlopen:{role}")
                break
    elif event == "import" and args and args[0] == "rocke_engine":
        _record("engine")


def _install_native_probe() -> None:
    """Wrap the comgr/HIP call chokepoint (channel 1)."""
    from rocke.runtime import _ctypes_bind

    cls = _ctypes_bind._LazyFn
    original = cls.__call__

    def probed(self, *args, **kwargs):
        resolver = getattr(self._lib_resolver, "__module__", "") or ""
        family = "comgr" if resolver.endswith("comgr") else (
            "hip" if resolver.endswith("hip_module") else "native"
        )
        _record(f"call:{family}")
        return original(self, *args, **kwargs)

    cls.__call__ = probed


def _mangle(nodeid: str) -> tuple[str, str]:
    """(classname, name) exactly as pytest's junitxml writes them, so the
    manifest joins onto the JUnit report. Mirrors _pytest.junitxml."""
    path, bracket, params = nodeid.partition("[")
    names = path.split("::")
    names[0] = names[0].replace(os.sep, ".").replace("/", ".")
    names[0] = re.sub(r"\.py$", "", names[0])
    names[-1] += bracket + params
    return ".".join(names[:-1]), names[-1]


# --- pytest hooks ----------------------------------------------------------


def pytest_configure(config):  # noqa: ANN001, ARG001
    sys.addaudithook(_audit)
    try:
        _install_native_probe()
    except Exception as exc:  # noqa: BLE001
        _S.install_errors.append(f"comgr/HIP call probe not installed: {exc!r}")


def pytest_collection_modifyitems(session, config, items):  # noqa: ANN001, ARG001
    for item in items:
        try:
            _S.module_of[item.nodeid] = str(item.path)
        except Exception:  # noqa: BLE001
            _S.module_of[item.nodeid] = ""


def pytest_runtest_logstart(nodeid, location):  # noqa: ANN001, ARG001
    _S.current.append(nodeid)


def pytest_runtest_logfinish(nodeid, location):  # noqa: ANN001, ARG001
    if _S.current:
        _S.current.pop()


def pytest_sessionfinish(session, exitstatus):  # noqa: ANN001, ARG001
    out = os.environ.get("ROCKE_RELEVANCE_OUT")
    if not out:
        return

    # A test failing before its compile step leaves no evidence of its own, but
    # its module's other tests still show whether the area drives the COD -- so
    # promote it to compiler-capable rather than writing it off as pure logic.
    touched_modules = {
        _S.module_of.get(nid, "")
        for nid, kinds in _S.evidence.items()
        if kinds and nid in _S.module_of
    }
    touched_modules.discard("")

    tests: dict[str, dict[str, object]] = {}
    for nid, module in _S.module_of.items():
        kinds = sorted(_S.evidence.get(nid, ()))
        if kinds:
            tier = TIER_COMPILER
        elif module in touched_modules:
            tier = TIER_CAPABLE
        else:
            tier = TIER_LOGIC
        classname, name = _mangle(nid)
        tests[f"{classname}\t{name}"] = {"tier": tier, "evidence": kinds}

    payload = {
        "version": MANIFEST_VERSION,
        "tests": tests,
        "install_errors": _S.install_errors,
    }
    tmp = f"{out}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, out)
