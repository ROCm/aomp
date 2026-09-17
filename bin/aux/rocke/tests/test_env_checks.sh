#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Tests for the environment checks that decide whether a lane may run at all.
# They are driven with stub tools, so the verdict is checked rather than the
# host: a check that reads the wrong variable still looks healthy on a host
# where both happen to be set.
#
# Run: bash bin/aux/rocke/tests/test_env_checks.sh

# shellcheck source-path=SCRIPTDIR
set -u

ScriptDir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HelperDir="${ScriptDir}/aux/rocke"
Passed=0
Failed=0

function check {  # <description> <expected> <actual>
  if [[ "${2}" == "${3}" ]]; then
    (( ++Passed ))
  else
    (( ++Failed ))
    echo "FAIL: ${1}: expected '${2}', got '${3}'"
  fi
}

# The module reports through the driver's row emitter and its failure path; both
# are stubbed so a verdict is observable.
function emitRockeResult { printf 'row|%s|%s\n' "${3}" "${4}"; }
function failSetup { printf 'fatal|%s\n' "${1}"; return 1; }

# shellcheck source=../rocke_env.source
. "${HelperDir}/rocke_env.source"

# A stub cmake of a given version, first on PATH.
function withCmake {  # <version|none> <command...>
  local Version="${1}"; shift
  local Dir; Dir="$(mktemp -d)"
  if [[ "${Version}" != none ]]; then
    printf '#!/bin/sh\necho "cmake version %s"\n' "${Version}" > "${Dir}/cmake"
    chmod +x "${Dir}/cmake"
  fi
  PATH="${Dir}:${PATH}" "$@"
  rm -rf "${Dir}"
}

# The version this check reads must be cmake's own: an earlier revision read an
# unrelated ROCm version, which passed wherever that happened to be set.
RocmVersion=""
check "a new enough cmake is accepted with no ROCm version known" \
  "" "$(withCmake 3.31.2 requireCmake)"

# shellcheck disable=SC2034 # read by the module under test
RocmVersion="10.2"
check "an old cmake is rejected and named, whatever the ROCm version" \
  "row|1|cmake 3.22 too old (need >= 3.25; set ROCKE_CMAKE_BIN)" \
  "$(withCmake 3.22.1 requireCmake)"

Absent="row|1|cmake not found (need >= 3.25 for rocKE's block();"
Absent+=" set ROCKE_CMAKE_BIN)"
check "a missing cmake is rejected" \
  "${Absent}" "$(PATH=/nonexistent withCmake none requireCmake 2>/dev/null)"

# The datalayout generation decides whether a torch build can serve as the
# numeric reference, so its boundary is pinned.
check "ROCm 7.1 is the plain p8 generation" \
  "plain-p8" "$(deriveDatalayoutGeneration 7.1)"
check "ROCm 7.2 is the indexed p8 generation" \
  "indexed-p8" "$(deriveDatalayoutGeneration 7.2)"
check "a bare major does not borrow itself as the minor" \
  "plain-p8" "$(deriveDatalayoutGeneration 7)"
check "a version that is not a number is refused" \
  "" "$(deriveDatalayoutGeneration nonsense 2>/dev/null || true)"

# A reset deletes and rebuilds a directory, so it must refuse anything that is
# not this suite's own interpreter.
NotAVenv="$(mktemp -d)"
ROCKE_VENV="${NotAVenv}" \
  check "a reset refuses a path that is not a virtual environment" \
  "ERROR: ${NotAVenv} is not a virtual environment: refusing to reset" \
  "$(ROCKE_VENV="${NotAVenv}" resetVenv 2>&1)"
check "and leaves it alone" "present" \
  "$([[ -d "${NotAVenv}" ]] && echo present || echo gone)"
rmdir "${NotAVenv}"

# The oldest release of a datalayout generation is what provisioning falls back
# to, since a toolchain's own ROCm often publishes no wheel.
check "an unreleased ROCm falls back to its generation's floor" \
  "7.2" "$(deriveGenerationFloor 10.2)"
check "a released ROCm of the same generation is its own floor" \
  "7.2" "$(deriveGenerationFloor 7.2)"
check "an older generation has its own floor" \
  "6.4" "$(deriveGenerationFloor 7.1)"
check "a version that is not a number has no floor" \
  "" "$(deriveGenerationFloor nonsense 2>/dev/null || true)"

# Provisioning order and consent. Both are checked with stubs: on a host that
# already has a usable torch, every path looks the same.
function validateTorch { [[ "${TorchUsable:-0}" == 1 ]]; }
function installRocmPackage {  # <package> <index>
  printf 'tried %s
' "${2}"
  [[ "${2}" == "${WorkingIndex:-}" ]] && TorchUsable=1
  [[ "${TorchUsable:-0}" == 1 ]]
}

TorchUsable=0
RocmVersion="10.2"
ROCKE_VENV="${TMPDIR:-/tmp}/a-venv-that-is-never-written"
ROCKE_PROVISION_TORCH=1
ROCKE_TORCH_INDEX_URL=""
WorkingIndex="https://download.pytorch.org/whl/rocm7.2"
Whl="https://download.pytorch.org/whl"
Want="tried ${Whl}/rocm10.2 tried ${Whl}/rocm7.2"
Want+=" row|Check|installed torch from ${Whl}/rocm7.2"
check "the toolchain's own index is tried before the floor" \
  "${Want}" "$(ensureTorch | paste -sd' ' -)"

TorchUsable=0
WorkingIndex="https://download.pytorch.org/whl/rocm10.2"
Want="tried ${Whl}/rocm10.2"
Want+=" row|Check|installed torch from ${Whl}/rocm10.2"
check "a working toolchain index stops the search" \
  "${Want}" "$(ensureTorch | paste -sd' ' -)"

TorchUsable=0
WorkingIndex=""
ROCKE_PROVISION_TORCH=0
check "the opt-out forbids the download" "" \
  "$(ensureTorch | paste -sd' ' -)"

ROCKE_PROVISION_TORCH=1
ROCKE_TORCH_INDEX_URL="file:///tmp/an-index"
WorkingIndex="file:///tmp/an-index"
TorchUsable=0
Want="tried file:///tmp/an-index"
Want+=" row|Check|installed torch from file:///tmp/an-index"
check "a stated index is used on its own" \
  "${Want}" "$(ensureTorch | paste -sd' ' -)"

TorchUsable=1
check "a usable torch is left alone" "" \
  "$(ensureTorch | paste -sd' ' -)"

echo "${Passed} passed, ${Failed} failed"
(( Failed == 0 ))
