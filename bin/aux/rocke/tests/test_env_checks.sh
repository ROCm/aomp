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

echo "${Passed} passed, ${Failed} failed"
(( Failed == 0 ))
