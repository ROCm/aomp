# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocKE <-> compiler toolchain interop smoke for a single GPU arch.
#
# Modes, all feeding rocKE-issued IR to the toolchain (see README's lane table):
#   codegen   : -> toolchain `clang` -> amdgcn relocatable object
#   comgr     : -> toolchain `libamd_comgr` -> HSACO, plus an on-device load+symbol
#               check when the local device matches the target arch
#   occupancy : -> HSACO, then its codegen resource footprint from the ELF notes
#               via rocke.benchmark.perf.occupancy; a spill is reported red
#
# One arch per process, so a fatal LLVM/comgr abort costs that arch's row rather
# than the whole lane. Result lines go to stdout, diagnostics to stderr, and the
# traceback only under ROCKE_DEBUG; always exits 0 so a failure is a data row.

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import rocke_comgr_version

from rocke_tiers import TIER_COMPILER
from rocke_result import emit as _result_emit

GROUP = "universal_gemm"
_OCCUPANCY_GROUP = "occupancy"
_SUBTEST_SUFFIX = ""


def _emit(subtest: str, status: int, message: str = "") -> None:
    # Every row here comes from driving the compiler toolchain, so a red one is
    # always a candidate compiler regression.
    _result_emit(GROUP, subtest + _SUBTEST_SUFFIX, status, message, TIER_COMPILER)


def _maybe_traceback() -> None:
    # The row already carries the concise reason, so this is debug-only noise.
    if os.environ.get("ROCKE_DEBUG"):
        traceback.print_exc()


def _is_rdna(arch: str) -> bool:
    """RDNA (gfx11xx/gfx12xx) lowers through WMMA at wave32; CDNA is wave64."""
    return arch.startswith(("gfx11", "gfx12"))


def _comgr_flavor(requested: str) -> str:
    """Resolve `auto` from the flavor the worker pinned for this toolchain.

    Validated against rocKE's own flavor list rather than a set written here, which
    would silently exclude a flavor rocKE adds.
    """
    if requested and requested != "auto":
        return requested

    from rocke.core.lower_llvm import LLVM_FLAVORS

    flavor = os.environ.get("ROCKE_COMGR_FLAVOR", "")
    if flavor not in LLVM_FLAVORS:
        raise RuntimeError(
            "auto comgr flavor requires ROCKE_COMGR_FLAVOR from the toolchain worker; "
            f"got {flavor!r}, expected one of {', '.join(LLVM_FLAVORS)}"
        )
    return flavor


def _lower_ir(arch: str, flavor: str) -> tuple[str, str]:
    """Lower the fixed smoke GEMM to IR, with the symbol the HSACO exports.

    rocKE mangles the spec into the kernel name, and the mangling is
    arch-dependent, so the device-load probe has to ask for the built name.
    """
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        build_universal_gemm,
    )
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    is_rdna = _is_rdna(arch)
    # compv3 is a CDNA/MFMA-only pipeline.
    pipeline = "wmma_v1" if is_rdna else "compv3"
    wave_size = 32 if is_rdna else 64
    warp_m, warp_n, warp_k = 2, 2, 1
    # gfx1250's WMMA catalog uses the 16x16x32 atom; the other swept targets
    # support 16x16x16.
    warp_tile_k = 32 if arch == "gfx1250" else 16
    # rocKE requires block_size == warp_m*warp_n*warp_k * wave_size.
    block_size = warp_m * warp_n * warp_k * wave_size
    spec = UniversalGemmSpec(
        name="cod_smoke",
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=warp_k,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=warp_tile_k,
        ),
        trait=TraitSpec(pipeline=pipeline, epilogue="default"),
        data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32"),
        wave_size=wave_size,
        block_size=block_size,
        batched=False,
    )
    kernel = build_universal_gemm(spec, arch=arch)
    return lower_kernel_to_llvm(kernel, arch=arch, llvm_flavor=flavor), kernel.name


def _patch(module: Any, name: str, value: Any) -> None:
    """Replace an existing module attribute, refusing to invent a new one.

    Every override here redirects rocKE away from system ROCm. Were the symbol
    renamed upstream, a plain assignment would still succeed and leave the probe
    quietly measuring the wrong toolchain.
    """
    if not hasattr(module, name):
        raise AttributeError(f"{module.__name__}.{name} is gone; probe needs an update")
    setattr(module, name, value)


def _build_hsaco(arch: str, flavor: str) -> tuple[bytes, str, str] | None:
    """(HSACO, kernel symbol, resolved flavor); None after emitting the red row."""
    try:
        flavor = _comgr_flavor(flavor)
    except Exception as exc:  # noqa: BLE001
        _emit(arch, 1, f"flavor resolution failed: {exc}")
        return None
    try:
        ir, symbol = _lower_ir(arch, flavor)
    except Exception as exc:  # noqa: BLE001
        _maybe_traceback()
        _emit(arch, 1, f"lower failed: {exc}")
        return None
    try:
        from rocke.runtime import comgr

        rocke_comgr_version.pinWhenUntrusted(comgr, flavor)
        hsaco, _timings = comgr.build_hsaco_from_llvm_ir(
            ir, isa=f"amdgcn-amd-amdhsa--{arch}"
        )
    except Exception as exc:  # noqa: BLE001
        _maybe_traceback()
        _emit(arch, 1, f"comgr compile failed ({flavor}): {exc}")
        return None
    if not hsaco:
        _emit(arch, 1, "comgr produced empty HSACO")
        return None
    print(f"comgr lib: {comgr.resolved_lib_path()}", file=sys.stderr)
    return hsaco, symbol, flavor


def _run_codegen(arch: str, flavor: str, clang: str, out_dir: str | None) -> None:
    try:
        ir, _ = _lower_ir(arch, flavor)
    except Exception as exc:  # noqa: BLE001 - report any lowering failure as a row
        _maybe_traceback()
        _emit(arch, 1, f"lower failed: {exc}")
        return

    # Falling back to a bare shared /tmp would write predictable {arch}.ll/.o
    # names another user on these lab machines could pre-plant as symlinks.
    if out_dir:
        out = Path(out_dir)
    else:
        # A caller-given --out is theirs to keep; one we invent is ours to remove.
        out = Path(tempfile.mkdtemp(prefix="rocke-smoke-"))
        atexit.register(shutil.rmtree, out, True)
    out.mkdir(parents=True, exist_ok=True)
    ll_path = out / f"{arch}.ll"
    obj_path = out / f"{arch}.o"
    ll_path.write_text(ir, encoding="utf-8")

    # The toolchain clang drives the same AMDGPU backend as a standalone llc and is
    # already hard-gated, so this needs no extra tool. `-x ir` marks .ll as IR.
    cmd = [
        clang, "-x", "ir", str(ll_path), "-c",
        "--target=amdgcn-amd-amdhsa", f"-mcpu={arch}", "-o", str(obj_path),
    ]
    print("+", " ".join(cmd), file=sys.stderr)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        _emit(arch, 1, f"clang did not run: {exc}")
        return
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        _emit(arch, 1, f"clang exit {proc.returncode}")
    elif not obj_path.exists() or obj_path.stat().st_size == 0:
        _emit(arch, 1, "clang produced no object")
    else:
        _emit(arch, 0, f"object {obj_path.stat().st_size} bytes")


def _run_comgr(arch: str, flavor: str) -> None:
    built = _build_hsaco(arch, flavor)
    if built is None:
        return
    hsaco, symbol, flavor = built
    _emit(arch, 0, f"HSACO {len(hsaco)} bytes ({flavor})")

    # Optional on-device load, only when this host's device matches the target.
    # Probing for a device is separate from loading onto one: a host with no
    # ROCm agent is not a failure of this host-only lane.
    try:
        from rocke.runtime.hip_module import get_device_arch
    except Exception as exc:  # noqa: BLE001 - the probe itself moved: say so
        _emit(f"{arch}-device-load", 1, f"cannot reach rocKE's device probe: {exc}")
        return
    try:
        device_arch = get_device_arch(0)
    except Exception:  # noqa: BLE001 - no runtime, no device: nothing to load here
        return
    if device_arch != arch:
        return
    try:
        from rocke.runtime.hip_module import Runtime

        mod = Runtime().load_module(hsaco)
        try:
            mod.get_function(symbol)
        finally:
            mod.unload()
        _emit(f"{arch}-device-load", 0, "loaded on device; kernel symbol found")
    except Exception as exc:  # noqa: BLE001 - a real load failure is a data row
        _emit(f"{arch}-device-load", 1, f"device load failed: {exc}")


def _run_occupancy(arch: str, flavor: str, readelf: str) -> None:
    """Compile with the toolchain's comgr, then report the HSACO's codegen resources."""
    built = _build_hsaco(arch, flavor)
    if built is None:
        return
    hsaco = built[0]

    try:
        from rocke.benchmark.perf import occupancy
    except Exception as exc:  # noqa: BLE001 - report API drift as a data row
        _emit(arch, 1, f"rocke.benchmark.perf.occupancy unavailable: {exc}")
        return

    try:
        # rocKE currently prefers /opt/rocm's readelf over PATH. Override its
        # private resolver so this toolchain probe uses the binary whose provenance
        # the driver validated. Assigning a missing attribute would succeed
        # silently and hand the probe back to system ROCm, so insist it is there.
        _patch(occupancy, "_readelf", lambda: readelf)
        res = occupancy.resources(hsaco, arch)
    except Exception as exc:  # noqa: BLE001 - probe failure is a data row
        _maybe_traceback()
        _emit(arch, 1, f"occupancy probe failed: {exc}")
        return
    if not res:
        _emit(arch, 1, "ELF notes unreadable (need a working llvm-readelf)")
        return
    # rocKE defaults every field to 0 when its ELF-note regexes do not match, so the
    # keys are always present and checking for them proves nothing. What proves the
    # notes were read is a value that cannot be zero for a compiled GEMM. This
    # catches a note format this llvm-readelf cannot parse at all; a rename of the
    # two spill notes alone would still read as "no spills", which is why the flavor
    # sweep and the drift gate -- not this row -- are what guard codegen changes.
    impossible = [k for k in ("vgpr", "sgpr", "occupancy") if not int(res.get(k) or 0)]
    if impossible:
        _emit(arch, 1, f"ELF notes not understood: {', '.join(impossible)} read as zero "
                       f"on a compiled kernel (need a matching llvm-readelf)")
        return
    vspill, sspill = int(res.get("vgpr_spill") or 0), int(res.get("sgpr_spill") or 0)
    summary = (
        f"vgpr={res.get('vgpr')} agpr={res.get('agpr')} sgpr={res.get('sgpr')} "
        f"lds={res.get('lds_bytes')}B spill={vspill}/{sspill} occ={res.get('occupancy')}"
    )
    if vspill or sspill:
        _emit(arch, 1, f"register spill on fixed smoke kernel: {summary}")
    else:
        _emit(arch, 0, summary)


def main() -> int:
    ap = argparse.ArgumentParser(description="rocKE<->toolchain interop smoke (single arch)")
    ap.add_argument("--mode", required=True, choices=("codegen", "comgr", "occupancy"))
    ap.add_argument("--arch", required=True)
    ap.add_argument("--flavor", default="llvm22")
    ap.add_argument("--clang", default="clang", help="toolchain clang (codegen mode)")
    ap.add_argument(
        "--readelf", default="llvm-readelf", help="toolchain llvm-readelf (occupancy mode)"
    )
    ap.add_argument("--out", help="object dir (codegen); default: a private temp dir")
    ap.add_argument(
        "--experimental",
        action="store_true",
        help="tag result rows as experimental (non-production arch)",
    )
    args = ap.parse_args()

    if args.experimental:
        global _SUBTEST_SUFFIX
        _SUBTEST_SUFFIX = " (experimental)"

    # Encode the compile path in the group so codegen and comgr rows stay distinct
    # in a consolidated suite. The extractor keys its by-area breakdown on the
    # first dotted component, so the area stays "universal_gemm".
    global GROUP
    if args.mode == "codegen":
        GROUP = f"{GROUP}.codegen"
        _run_codegen(args.arch, args.flavor, args.clang, args.out)
    elif args.mode == "occupancy":
        GROUP = _OCCUPANCY_GROUP
        _run_occupancy(args.arch, args.flavor, args.readelf)
    else:
        GROUP = f"{GROUP}.comgr"
        _run_comgr(args.arch, args.flavor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
