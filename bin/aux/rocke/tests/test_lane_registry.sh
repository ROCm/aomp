#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The lane registry decides what runs, what a row means and what provenance gets
# proved, so a table that is wrong has to stop the run rather than produce rows.
# These tests state each way it can be wrong and check it is refused.
#
# Run: bash bin/aux/rocke/tests/test_lane_registry.sh

# Source directives below resolve from this script's directory, not the caller's.
# shellcheck source-path=SCRIPTDIR
set -u
Here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Pass=0
Fail=0

# The module needs two things from the driver: somewhere to report, and the arch
# and flavor lists the run-time floors are sized from.
function failSetup { echo "FATAL: ${1}"; exit 9; }
function emitRockeResult { :; }
export ROCKE_CI_ARCHES="gfx950 gfx942"
export ROCKE_CI_ARCHES_EXPERIMENTAL="gfx90a"
export ROCKE_ENGINE_FLAVORS="llvm20 llvm22"

# shellcheck source=../rocke_lanes.source
. "${Here}/../rocke_lanes.source"
# Stand-ins for the lane bodies, which live in the same module but are not what is
# under test here.
for Lane in $(listLaneNames); do eval "function $(deriveLaneHandler "${Lane}") { :; }"; done

function check {  # <description> <expected> <actual>
  if [[ "${2}" == "${3}" ]]; then
    Pass=$((Pass + 1))
  else
    Fail=$((Fail + 1))
    printf 'FAIL  %s\n      expected [%s]\n      got      [%s]\n' "${1}" "${2}" "${3}"
  fi
}

# Run assertLaneRegistry over a registry of our own and report what it said, so each
# case below states a malformed table and the complaint it must produce.
function verdict {  # <registry row>...
  ( LaneRegistry=("$@")
    LaneOrder=()
    for Row in "${LaneRegistry[@]}"; do LaneOrder+=("${Row%%|*}"); done
    assertLaneRegistry 2>&1 | sed -n 's/.*not usable: *//p' ) | tr -d '\n'
}

Good="ctest|rocKE|compiler|3|"

check "a valid row is accepted" "" "$(verdict "${Good}")"
check "a missing field is refused" "fields:ctest" "$(verdict "ctest|rocKE|compiler|3")"
check "an extra field is refused" "fields:ctest" "$(verdict "ctest|rocKE|compiler|3||x")"
check "an empty name is refused" "name:<empty>" "$(verdict "|rocKE|compiler|3|")"
check "a duplicate lane is refused" "duplicate:ctest" "$(verdict "${Good}" "${Good}")"
check "an unknown origin is refused" "origin:ctest" "$(verdict "ctest|elsewhere|compiler|3|")"
check "an unknown relevance is refused" "relevance:ctest" "$(verdict "ctest|rocKE|typo|3|")"
check "an empty floor is refused" "floor:ctest" "$(verdict "ctest|rocKE|compiler||")"
check "a non-numeric floor is refused" "floor:ctest" "$(verdict "ctest|rocKE|compiler|many|")"
check "an unknown tool is refused" "tool:ctest:nosuch" "$(verdict "ctest|rocKE|compiler|3|nosuch")"
check "a lane with no body is refused" "handler:nosuchlane" \
  "$(verdict "nosuchlane|rocKE|compiler|3|")"

# The handler is derived, not stored, so no table can name another lane's body.
check "handler of engine" "runEngineLane" "$(deriveLaneHandler engine)"
check "handler of codegen" "runCodegenLane" "$(deriveLaneHandler codegen)"
check "handler of gpu-numeric" "runGpuNumericLane" "$(deriveLaneHandler gpu-numeric)"
check "every registered lane has a body defined" "" \
  "$(for L in $(listLaneNames); do declare -F "$(deriveLaneHandler "${L}")" >/dev/null || echo "${L}"; done)"

# Floors sized at run time follow the lists they are sized from.
check "the engine floor counts flavors" "2" "$(readLaneRowFloor engine)"
check "an arch floor counts both sweeps" "3" "$(readLaneRowFloor comgr)"
check "an unregistered lane has no floor" "?" "$(readLaneRowFloor nosuchlane)"
check "an unregistered lane has no relevance" "unregistered" "$(readLaneRelevance nosuchlane)"
check "an empty tools field yields nothing" "" "$(readLaneHardTools ctest)"

printf '%s passed, %s failed\n' "${Pass}" "${Fail}"
(( Fail == 0 ))
