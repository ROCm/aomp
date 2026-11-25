# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocKE reads the ROCm version of the comgr it loads by climbing to an install root
# that ships a version file, and falls back to the system ROCm when it finds none.
# A compiler-only install has no such file, so rocKE then gates its features on a
# foreign number: it refuses architectures the compiler toolchain supports.
#
# The driver establishes the build's own ROCm version and says whether the number
# rocKE would read can be believed. Where it cannot, this module tells rocKE the
# build's number instead. Both the compile sweeps and the test session use it, so
# the policy lives here rather than in either of them.
from __future__ import annotations

import os
from typing import Any

TRUSTED_ENV = "ROCKE_COMGR_VERSION_TRUSTED"
BUILD_VERSION_ENV = "ROCKE_BUILD_ROCM_VERSION"


def buildVersion() -> tuple[int, int] | None:
    """The ROCm version the driver established for this build, as (major, minor)."""
    raw = os.environ.get(BUILD_VERSION_ENV, "")
    parts = raw.split(".")[:2]
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return int(parts[0]), int(parts[1])
    return None


def versionForFlavor(flavor: str) -> tuple[int, int] | None:
    """A ROCm version rocKE maps to ``flavor``, read off its own ladder.

    Inverting the ladder by hand puts the version in the wrong datalayout
    generation, and rocKE's flavor guard then rejects the IR it was asked to lower.
    """
    from rocke.core.lower_llvm import _ROCM_FLAVOR_LADDER

    version = next((v for v, f in _ROCM_FLAVOR_LADDER if f == flavor), None)
    if version is not None:
        return version
    if not _ROCM_FLAVOR_LADDER:
        return None
    # Below the oldest ladder row: the flavor that predates every entry.
    oldest = _ROCM_FLAVOR_LADDER[-1][0]
    return (oldest[0], oldest[1] - 1) if oldest[1] else (oldest[0] - 1, 0)


def pin(comgr: Any, version: tuple[int, int]) -> None:
    """Make rocKE report ``version`` for the comgr it loads."""
    name = "resolved_lib_rocm_version"
    if not hasattr(comgr, name):
        raise AttributeError(f"{comgr.__name__}.{name} is gone; this helper needs an update")
    setattr(comgr, name, lambda: version)


def pinWhenUntrusted(comgr: Any, flavor: str = "") -> tuple[int, int] | None:
    """Pin the build's ROCm version when the one rocKE reads cannot be believed.

    Returns the version pinned, or None when nothing was changed. Pinning a
    trusted install would also satisfy rocKE's IR-flavor guard and cover up a
    genuine comgr-versus-clang mismatch, which is a signal this suite reports.
    """
    if os.environ.get(TRUSTED_ENV) != "0":
        return None
    version = buildVersion()
    if version is None and flavor:
        version = versionForFlavor(flavor)
    if version is None:
        return None
    pin(comgr, version)
    return version
