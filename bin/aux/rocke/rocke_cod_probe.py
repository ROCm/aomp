#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Report the comgr library rocKE will actually load, so run_rocke.sh's hygiene
# gate can prove it resolves inside the COD. Prints one line:
#
#   <chosen-flavor> <vintage-flavor> <rocm-vintage> <interface-version> <lib-path>
#
# Takes the p8 datalayout generation the COD clang emits ("plain"/"indexed"/
# "unknown") as its one argument, and decides the flavor here so that all knowledge
# of rocKE's flavors stays on this side of the boundary: which flavors exist, which
# ROCm each belongs to, and which datalayout generation each emits are all read from
# rocKE, so a flavor it adds needs no edit anywhere.
#
# vintage-flavor is what rocKE's own release ladder derives from the loaded comgr's
# ROCm vintage. chosen-flavor is what the worker should pin: the clang's generation
# decides, because a datalayout cannot leak from an unrelated tree while a ROCm
# version can (a COD without .info/version makes rocKE read the system /opt/rocm).
# Inside that generation the vintage is believed when it agrees, since it is the only
# thing that can tell two flavors of the same generation apart; when it disagrees the
# oldest flavor of the clang's generation is used, and the two fields differing is
# the worker's signal to warn.
#
# interface-version comes from the lib's own amd_comgr_get_version (vintage-proof).
# Any field is "?"/"UNRESOLVED" when it cannot be determined.

from __future__ import annotations

import ctypes
import sys

# plain | indexed | unknown, from the COD clang's own target datalayout.
shape = sys.argv[1] if len(sys.argv) > 1 else "unknown"

# Two questions, two guards. Folded into one try, a renamed flavor helper also made
# `path` read UNRESOLVED, so the worker's hygiene gate reported the comgr as MISSING
# and refused to run every lane -- blaming the COD install for an upstream rename.
try:
    from rocke.runtime.comgr import resolved_lib_path, resolved_lib_rocm_version

    path = resolved_lib_path() or "UNRESOLVED"
    ver = resolved_lib_rocm_version()
    rocm = f"{ver[0]}.{ver[1]}" if ver else "?"
except Exception:
    path, rocm, ver = "UNRESOLVED", "?", None

try:
    from rocke.core.lower_llvm import _flavor_for_rocm

    # rocKE's ladder, not a copy of it: it gained llvm23 at ROCm 7.13 while a copy
    # here still said llvm22. No vintage means no basis for a flavor; claiming one
    # would make the worker warn about a mismatch it never measured.
    flavor = "?" if ver is None else _flavor_for_rocm(*ver)
except Exception:
    flavor = "?"

chosen = flavor
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
        # rocKE partitions its flavors by generation and tests that the partition is
        # exhaustive, so this asks instead of matching flavor names.
        same_gen = [f for f in LLVM_FLAVORS if _datalayout_kind_for_flavor(f) == want]
        if same_gen:
            chosen = flavor if flavor in same_gen else same_gen[0]
except Exception:
    pass

iface = "?"
try:
    fn = ctypes.CDLL(path).amd_comgr_get_version
    fn.argtypes = [ctypes.POINTER(ctypes.c_size_t)] * 2
    major, minor = ctypes.c_size_t(), ctypes.c_size_t()
    fn(ctypes.byref(major), ctypes.byref(minor))
    iface = f"{major.value}.{minor.value}"
except Exception:
    pass

print(chosen, flavor, rocm, iface, path)
