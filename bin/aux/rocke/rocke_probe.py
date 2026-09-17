#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Facts about the environment that the driver cannot ask for in shell: the comgr
# rocKE would load, the visible device, the installed packages, and the test
# dependencies rocKE declares. One module so these stay together and the shell
# stays free of embedded Python.
#
#   rocke_probe.py comgr <plain|indexed|unknown>   -> flavor vintage-flavor rocm iface lib
#   rocke_probe.py device                          -> gfxNNN | none | error: <Type>
#   rocke_probe.py packages                        -> "<name> <version> <location>" lines
#   rocke_probe.py project-deps <pyproject.toml>   -> one PEP 508 requirement per line
#
# Every probe answers rather than raises: an unanswerable question is reported in
# the value ("?", "UNRESOLVED", "error: ..."), because the caller turns that into a
# row and a crash here would end the lane instead.

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata as metadata
import pathlib
import re
import sys

# Packages whose version can move a verdict with no compiler change: pytest decides
# what ran, pytest-subtests the shape of the report the CI reconciles against, numpy
# and torch are the numeric references, pybind11 builds the engine extension.
REPORTED_PACKAGES = ("pytest", "pytest-subtests", "numpy", "pybind11", "torch")


def probeComgr(shape: str) -> str:
    """The comgr rocKE resolves, and the IR flavor to pin for it.

    Which flavors exist, which ROCm each belongs to and which datalayout each emits
    are read from rocKE, so a flavor it adds needs no edit here.

    Two guards, not one: folded together, a renamed flavor helper also made the path
    read UNRESOLVED, and the hygiene gate then blamed the COD install for it.
    """
    try:
        from rocke.runtime.comgr import resolved_lib_path, resolved_lib_rocm_version

        path = resolved_lib_path() or "UNRESOLVED"
        version = resolved_lib_rocm_version()
        rocm = f"{version[0]}.{version[1]}" if version else "?"
    except Exception:  # noqa: BLE001
        path, rocm, version = "UNRESOLVED", "?", None

    try:
        from rocke.core.lower_llvm import _flavor_for_rocm

        vintageFlavor = "?" if version is None else _flavor_for_rocm(*version)
    except Exception:  # noqa: BLE001
        vintageFlavor = "?"

    # The clang's datalayout generation decides, because it cannot leak from another
    # tree while a ROCm version can. Inside that generation the vintage is believed
    # when it agrees; when it does not, the caller warns about the split.
    chosen = vintageFlavor
    try:
        from rocke.core.lower_llvm import (
            LLVM_FLAVORS,
            LlvmDatalayoutKind,
            _datalayout_kind_for_flavor,
        )

        want = {
            "plain": LlvmDatalayoutKind.P8_PLAIN,
            "indexed": LlvmDatalayoutKind.P8_INDEXED,
        }.get(shape)
        if want is not None:
            sameGeneration = [
                f for f in LLVM_FLAVORS if _datalayout_kind_for_flavor(f) == want
            ]
            if sameGeneration:
                chosen = (
                    vintageFlavor
                    if vintageFlavor in sameGeneration
                    else sameGeneration[0]
                )
    except Exception:  # noqa: BLE001
        pass

    interface = "?"
    try:  # from the lib itself, so it holds whatever the vintage metadata says
        getVersion = ctypes.CDLL(path).amd_comgr_get_version
        getVersion.argtypes = [ctypes.POINTER(ctypes.c_size_t)] * 2
        major, minor = ctypes.c_size_t(), ctypes.c_size_t()
        getVersion(ctypes.byref(major), ctypes.byref(minor))
        interface = f"{major.value}.{minor.value}"
    except Exception:  # noqa: BLE001
        pass

    return f"{chosen} {vintageFlavor} {rocm} {interface} {path}"


def probeDevice() -> str:
    """The arch of device 0, or why it could not be asked.

    "no GPU here" and "the question failed" are different answers: collapsing them
    lets a rocKE rename retire the only lane that can catch a miscompile, silently.
    """
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0) or "none"
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}: {exc}".replace("\n", "; ")


def probePackages() -> str:
    lines = []
    for name in REPORTED_PACKAGES:
        try:
            version = metadata.version(name)
        except Exception:  # noqa: BLE001
            lines.append(f"{name:16} absent")
            continue
        try:
            module = __import__(name.replace("-", "_"))
            where = str(pathlib.Path(getattr(module, "__file__", "") or "").parent)
        except Exception:  # noqa: BLE001
            where = "(not importable)"
        detail = ""
        if name == "torch":
            try:
                import torch

                detail = (
                    f" hip {torch.version.hip}"
                    if torch.version.hip
                    else " not a ROCm build"
                )
            except Exception:  # noqa: BLE001
                detail = " (not importable)"
        lines.append(f"{name:16} {version}{detail}  {where}")
    return "\n".join(lines)


def projectDeps(path: str) -> str:
    """rocKE's declared dev requirements, whole specs including version bounds.

    Reduced to bare names, an installed pybind11 2.x satisfied rocKE's `pybind11>=3.0`
    and pip did nothing, so the bound they had just raised was never applied.
    """
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        sys.exit(f"cannot read {path}: {exc}")

    specs: list[str] = []
    try:
        import tomllib

        specs = (
            tomllib.loads(text)
            .get("project", {})
            .get("optional-dependencies", {})
            .get("dev", [])
        )
    except ModuleNotFoundError:  # 3.10 has no tomllib: read the one table we need
        # Anchored in that table, because a bare `dev = [` also matches a PEP 735
        # [dependency-groups] list, which is a different set of requirements. The
        # closing bracket must start a line, or a spec carrying its own extras
        # ("pandas[performance]") ends the list early.
        table = re.search(
            r"^\[project\.optional-dependencies\](.*?)(?=^\[|\Z)", text, re.M | re.S
        )
        block = (
            re.search(r"^\s*dev\s*=\s*\[(.*?)^\s*\]", table.group(1), re.M | re.S)
            if table
            else None
        )
        if block:
            specs = re.findall(r'"([^"]+)"', block.group(1))
    except (ValueError, TypeError) as exc:
        sys.exit(f"cannot parse {path}: {exc}")

    specs = [s.strip() for s in dict.fromkeys(specs) if s.strip()]
    if not specs:
        sys.exit(f"no [project.optional-dependencies].dev entries in {path}")
    return "\n".join(specs)


def main() -> int:
    parser = argparse.ArgumentParser(description="environment probes for run_rocke.sh")
    sub = parser.add_subparsers(dest="probe", required=True)
    comgr = sub.add_parser("comgr")
    comgr.add_argument("shape", nargs="?", default="unknown",
                       choices=("plain", "indexed", "unknown"))
    sub.add_parser("device")
    sub.add_parser("packages")
    deps = sub.add_parser("project-deps")
    deps.add_argument("pyproject")
    args = parser.parse_args()

    if args.probe == "comgr":
        print(probeComgr(args.shape))
    elif args.probe == "device":
        print(probeDevice())
    elif args.probe == "packages":
        print(probePackages())
    else:
        print(projectDeps(args.pyproject))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
