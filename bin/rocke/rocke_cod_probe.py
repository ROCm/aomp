#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Report the comgr library rocKE will actually load, so run_rocke.sh's hygiene
# gate can prove it resolves inside the COD. Prints one line:
#
#   <ir-flavor> <rocm-vintage> <interface-version> <lib-path>
#
# ir-flavor is what the loaded comgr's ROCm vintage implies (>= 7.2 -> llvm22,
# else llvm20); interface-version comes from the lib's own amd_comgr_get_version
# (vintage-proof). Any field is "?"/"UNRESOLVED" when it cannot be determined.

from __future__ import annotations

import ctypes

try:
    from rocke.runtime.comgr import resolved_lib_path, resolved_lib_rocm_version

    path = resolved_lib_path() or "UNRESOLVED"
    ver = resolved_lib_rocm_version()
    rocm = f"{ver[0]}.{ver[1]}" if ver else "?"
    # No vintage means no basis for a flavor; claiming llvm20 would make
    # run_rocke.sh warn about a mismatch it never measured.
    flavor = "?" if ver is None else ("llvm22" if ver >= (7, 2) else "llvm20")
except Exception:
    path, rocm, flavor = "UNRESOLVED", "?", "?"

iface = "?"
try:
    fn = ctypes.CDLL(path).amd_comgr_get_version
    fn.argtypes = [ctypes.POINTER(ctypes.c_size_t)] * 2
    major, minor = ctypes.c_size_t(), ctypes.c_size_t()
    fn(ctypes.byref(major), ctypes.byref(minor))
    iface = f"{major.value}.{minor.value}"
except Exception:
    pass

print(flavor, rocm, iface, path)
