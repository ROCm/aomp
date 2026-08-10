#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocKE driver: run one lane against the compiler-of-the-day (COD), emit
# canonical "ROCKE_RESULT|group|subtest|status|message|relevance" lines and
# close with a short summary. Self-contained, so an engineer can run it by hand;
# the nightly drives the same lanes. The worker modules it runs are in bin/rocke;
# the ROCKE_* knobs default just below. The
# CI-side extractor that turns the rows into dashboard rows, and the full
# documentation, live in the apps repo under openmp-ci/rocKE (extract-rocke.sh,
# README.md).
#
# Usage: run_rocke.sh [-r] [-u] <lane>; -h lists the lanes and the common knobs.
# 'all' runs every lane in one process for a single consolidated report/mail
# (see USER-run-rocKE-all); schedule the per-lane wrappers for one report each.
# The -r/-u flags mirror CK's; the nightly wrappers set both by env instead.

set -u

# The lanes this driver knows, in the order 'all' runs them: the cheap host-only
# gates first so a broken COD is reported in seconds, the ~1000-row pytest lane
# next, then the on-device lane, and perf last because a register-spill verdict is
# only worth reading once the kernels it measures are known to compile.
# Single source of truth: the stage check, the help text and the 'all' lane list
# all derive from it, so none of them can be updated without the others.
LaneOrder=(engine ctest pytest cod-codegen cod-comgr gpu-numeric perf)
Lanes="all|$(IFS="|"; printf '%s' "${LaneOrder[*]}")"

function printUsage {
  cat <<EOF
usage: ${0##*/} [-r] [-u] <${Lanes//|/ | }>

Runs one rocKE lane against the compiler-of-the-day and prints ROCKE_RESULT rows;
'all' runs every lane and adds a pass/total tally per lane. A failing test is a
result row, not a driver error, so a red row does not change the exit status: read
the closing summary or the rows, not \$?. ('all' does exit non-zero if a lane died
or produced no rows at all -- that is a broken run, not a test result.)

  -r  rebuild each lane from a clean build dir   (ROCKE_REBUILD=1; off by default)
  -u  refresh the shared rocm-libraries checkout (ROCKE_UPDATE_REPO=1; off by default)
  -h  this help

Common knobs (every ROCKE_* default is set together at the top of this script;
the lane table and the full list are in openmp-ci/rocKE/README.md):
  AOMP=<llvm dir>            COD compiler under test
  ROCKE_ALL_LANES='...'      lanes 'all' runs, in order
  ROCKE_CI_ARCHES='...'      arch sweep for the cod-*/perf lanes
  ROCKE_TOP=<dir>            rocKE platform checkout to test (else one is cloned)
  ROCKE_CI_BUILD_ROOT=<dir>  out-of-tree build root
  ROCKE_VENV=<dir>           the only interpreter this script may install into
  ROCKE_DEBUG=1              full Python tracebacks from the cod-*/perf lanes
EOF
}

# Leading -r/-u are sugar for the ROCKE_REBUILD / ROCKE_UPDATE_REPO gates the
# worker reads below; the 'all' fork exports them, so lane children inherit the
# same choice.
while [[ "${1:-}" == -?* ]]; do
  case "$1" in
    -r) export ROCKE_REBUILD=1 ;;
    -u) export ROCKE_UPDATE_REPO=1 ;;
    -h|--help) printUsage; exit 0 ;;
    --) shift; break ;;
    *)  echo "unknown option: $1"; printUsage; exit 2 ;;
  esac
  shift
done

Stage="${1:-}"
(( $# > 1 )) && { echo "ERROR: unexpected argument: ${2}" >&2; printUsage; exit 2; }
ScriptDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The worker modules keep their rocke_ prefix: they go on the PYTHONPATH of
# rocKE's own test session, where a plain result.py would shadow the project's.
HelperDir="${ScriptDir}/rocke"
InheritedPythonPath="${PYTHONPATH:-}"
if [[ -z "${Stage}" ]]; then
  printUsage; exit 2
elif [[ "|${Lanes}|" != *"|${Stage}|"* ]]; then
  echo "unknown stage: ${Stage}"; printUsage; exit 2
fi

# Compiler-of-the-day (COD) toolchain. A single knob, AOMP, selects the compiler
# under test; every other tool (comgr, HIP runtime, hipcc, device-libs) is derived
# from the install that ships it, so no lane can silently fall back to a system
# ROCm (/opt/rocm) or /usr toolchain. The assertCodToolchain gate below proves it.
: "${AOMP:=/COD/LATEST/aomp/llvm}"
AompInput="${AOMP}"
AOMP="$(realpath -m "${AompInput}")"
export AOMP

# Walk up from the resolved llvm dir to the nearest ancestor shipping comgr or
# hipcc. Key on those, not on include/hip or amdgcn/bitcode: some packagings put
# headers and device-libs under llvm/, so keying on those would stop one level
# below the real root and let comgr fall back to a system /opt/rocm.
function resolveRocmRoot {
  local Dir="${1}" Start="${1}" _Hop
  for _Hop in 0 1 2 3; do
    if [[ -e "${Dir}/lib/libamd_comgr.so" || -x "${Dir}/bin/hipcc" ]]; then
      echo "${Dir}"; return 0
    fi
    Dir="$(realpath -m "${Dir}/..")"
  done
  realpath -m "$(dirname "${Start}")"  # give up; prefix check + hygiene flag it
}

# ROCM_PATH (the house-standard knob) may override the derived root, but only
# while realpath(ROCM_PATH) is a prefix of realpath(AOMP): otherwise a stray
# ambient `export ROCM_PATH=/opt/rocm` would hijack the root and wave a system
# ROCm through the hygiene gate as [COD].
DerivedRoot="$(resolveRocmRoot "${AOMP}")"
if [[ -n "${ROCM_PATH:-}" ]]; then
  EnvRoot="$(realpath -m "${ROCM_PATH}")"
  if [[ "${AOMP}/" == "${EnvRoot}/"* ]]; then
    RocmRoot="${EnvRoot}"; RocmRootSource="from env ROCM_PATH"
  else
    echo "WARNING: ignoring ROCM_PATH=${EnvRoot} -- not a prefix of AOMP=${AOMP}"
    echo "         (looks like an ambient/system ROCm). Using the AOMP-derived root."
    echo "         To force a specific root, point ROCM_PATH at an ancestor of AOMP."
    RocmRoot="${DerivedRoot}"; RocmRootSource="derived from AOMP (ROCM_PATH ignored)"
  fi
else
  RocmRoot="${DerivedRoot}"; RocmRootSource="derived from AOMP"
fi

export ROCM_PATH="${RocmRoot}"
export HIP_PATH="${RocmRoot}"
# Pin hipcc's clang to the COD llvm so the HIP path can never pick a system clang.
export HIP_CLANG_PATH="${AOMP}/bin"
export ROCKE_COMGR_LIB="${ROCKE_COMGR_LIB:-${RocmRoot}/lib/libamd_comgr.so}"
export ROCKE_HIP_LIB="${ROCKE_HIP_LIB:-${RocmRoot}/lib/libamdhip64.so}"
export CC="${AOMP}/bin/clang"
export CXX="${AOMP}/bin/clang++"
# rocKE compiles its C++ engine by invoking the plain name `c++` (see
# tests/instances/differential/run_diff.py), and no COD ships that name -- it shipped
# clang++ only -- so `c++` resolved to the system compiler and the byte-identity lane
# built rocKE's engine with /usr/bin/c++ while its rows claimed COD provenance. A
# shim directory ahead of everything makes that name mean the COD's clang++, which is
# what the lane is supposed to be measuring. Anything else upstream invokes by a bare
# name can be added here rather than patched into rocKE.
# installCodShim fills this in; PATH may name a directory that does not exist yet.
CodShim="${TMPDIR:-/tmp}/rocke-cod-shim-$$"
# COD llvm tools first, then the shim, then the install bin (hipcc, rocprofv3).
export PATH="${AOMP}/bin:${CodShim}:${RocmRoot}/bin:${PATH}"
# AOMP compiler runtime first (libomp/libomptarget), then the pinned ROCm
# runtime (libamdhip64/libhsa). Guard the tail: an unset var must not leave a
# trailing ':' -- an empty entry means CWD, which would breach COD isolation.
export LD_LIBRARY_PATH="${AOMP}/lib:${RocmRoot}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# rocKE's CMakeLists uses block(), which needs CMake >= 3.25; prefer a modern
# local cmake when the distro one is older.
: "${ROCKE_CMAKE_BIN:=${HOME}/local/cmake/bin}"
[[ -x "${ROCKE_CMAKE_BIN}/cmake" ]] && export PATH="${ROCKE_CMAKE_BIN}:${PATH}"

: "${AOMP_REPOS_TEST:=${HOME}/git/aomp-test}"
: "${ROCKE_TOP:=${AOMP_REPOS_TEST}/composable-kernels/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/platform}"
# -s keeps this in ROCKE_TOP's path namespace. Resolving symlinks can put the
# library test root under a different prefix (/work/... vs /home/...), which makes
# pytest root its collection tree at / and scan shared parents like /work,
# aborting on the first unreadable entry there.
ROCKE_PROJECT_ROOT="$(realpath -m -s "${ROCKE_TOP}/..")"
# Path from the shared rocm-libraries repo root down to the rocKE platform dir.
ROCKE_TOP_SUFFIX="/dnn-providers/hip-kernel-provider/rocke/platform"
ROCKE_REPO_ROOT="${ROCKE_TOP%"${ROCKE_TOP_SUFFIX}"}"
: "${ROCKE_VENV:=${HOME}/.local/rocKE-venv}"
: "${ROCKE_CODEGEN_FLAVOR:=auto}"
: "${ROCKE_COMGR_FLAVOR:=auto}"
: "${ROCKE_CI_ARCHES:=gfx950 gfx942 gfx1151 gfx1201}"
# Experimental arches, appended to the COD compile lanes only (not part of the
# production sweep); each compiles through the COD and exercises the on-device
# HSACO load when the runner matches. Set empty to disable.
: "${ROCKE_CI_ARCHES_EXPERIMENTAL=gfx90a gfx1250}"
# Default to every flavor rocKE declares (it publishes the list; asking keeps a new
# one from going unswept, which is how llvm23 arrived unnoticed). Falls back to the
# pair we know if the list cannot be read -- resolved late, in stageEngine, since
# rocKE is not importable this early.
: "${ROCKE_ENGINE_FLAVORS:=auto}"
# Families the byte-identity gate must cover before its pass means anything;
# rocKE ships 68 today, so this only fires on a collapse.
: "${ROCKE_MIN_GATE_FAMILIES:=20}"
# Sibling of the rocm-libraries checkout, matching the ck-src/ck-build layout
# beside it. Off /tmp, so no reaper can drop the tree or its root marker.
if [[ "${ROCKE_REPO_ROOT}" == "${ROCKE_TOP}" && -z "${ROCKE_CI_BUILD_ROOT:-}" ]]; then
  # A ROCKE_TOP outside a rocm-libraries checkout leaves nothing to sit beside,
  # so the derived root would land inside the engineer's own tree.
  echo "WARNING: ROCKE_TOP is not inside a rocm-libraries checkout; the build root"
  echo "         will be derived next to it. Set ROCKE_CI_BUILD_ROOT to choose one."
fi
: "${ROCKE_CI_BUILD_ROOT:=${ROCKE_REPO_ROOT%/*}/rocke-build}"
ROCKE_CI_BUILD_ROOT="$(realpath -m "${ROCKE_CI_BUILD_ROOT}")"
# Lanes 'all' runs, in LaneOrder (see the top of this file for why that order).
# Override to scope a run, e.g. ROCKE_ALL_LANES='engine ctest'.
: "${ROCKE_ALL_LANES:=${LaneOrder[*]}}"

# Run-shape knobs. The nightly wrappers pin both gates to 1; run by hand they
# default off, so a rerun neither wipes the lane build dir nor touches the shared
# checkout, and -r/-u opt in exactly as they do for CK's driver.
: "${ROCKE_REBUILD:=0}"
: "${ROCKE_UPDATE_REPO:=0}"
: "${ROCKE_SETUP_VENV:=1}"
: "${ROCKE_REPO_URL:=https://github.com/ROCm/rocm-libraries.git}"
: "${ROCKE_REPO_BRANCH:=develop}"
: "${ROCKE_TORCH_INDEX_URL:=}"
# rocKE uses these without declaring them in its dev extras.
: "${ROCKE_EXTRA_TEST_DEPS:=pyarrow}"
# Identity of this run, inherited by the child lanes of 'all'. A bare PID would do
# for that, but it is also persisted as the engine-extension rebuild stamp, and PIDs
# repeat: a night that drew a previous run's PID would silently skip a demanded
# rebuild and test yesterday's artefact. Seconds since the epoch cannot repeat.
# Only a lane of this run may inherit it; an ambient value from an unrelated shell
# would otherwise be adopted as this run's identity and cancel a demanded rebuild,
# which is the failure the id exists to prevent.
if [[ -z "${ROCKE_INTERNAL_PARENT_PID:-}" || "${ROCKE_INTERNAL_PARENT_PID}" != "${PPID}" ]]; then
  ROCKE_RUN_ID=""
fi
export ROCKE_RUN_ID="${ROCKE_RUN_ID:-$(date +%s)-${BASHPID}}"

# A concurrent run holds the source lock for its whole duration -- deliberately,
# so the tree cannot change under a test in flight -- so allow a nightly's worth
# of waiting, then fail with a red row rather than hang.
: "${ROCKE_LOCK_WAIT:=3600}"
# Reaches an arithmetic context, where bash would evaluate anything it is given.
[[ "${ROCKE_LOCK_WAIT}" =~ ^[0-9]+$ ]] \
  || { echo "ERROR: ROCKE_LOCK_WAIT must be a whole number of seconds" >&2; exit 2; }

# The interop lanes want deterministic reference IR: the C++ engine is
# byte-identical but not built here, so use the Python backend and skip the
# noisy fallback warning.
export ROCKE_BACKEND="${ROCKE_BACKEND:-python}"
export PYTHONPATH="${ROCKE_TOP}/python:${ROCKE_PROJECT_ROOT}/library"
export PYTHONPATH+="${InheritedPythonPath:+:${InheritedPythonPath}}"

BuildRoot="$(realpath -m "${ROCKE_CI_BUILD_ROOT}/${Stage}")"
PyBin=""

# Every row emitted anywhere in the run is copied here (by this function, and by
# rocke_result.py via ROCKE_ROW_LOG) so the closing summary can tally the run
# without capturing its own stdout. Empty until a lane opens one.
RowLog=""

# Canonical result line consumed by extract-rocke.sh; '|' is the field
# separator, so strip it from caller-supplied fields. Relevance says how a red
# row should be triaged (see rocke_tiers.py); it defaults to this CI's own
# plumbing because that is what the bash-side rows are.
function rockeResult {
  local Group="${1//|/ }" Subtest="${2//|/ }" Status="${3}" Message="${4//|/ }"
  local Relevance="${5:-harness}" Row
  Row="ROCKE_RESULT|${Group}|${Subtest}|${Status}|${Message}|${Relevance//|/ }"
  echo "${Row}"
  if [[ -n "${RowLog}" ]]; then printf '%s\n' "${Row}" >> "${RowLog}"; fi
}

# A setup failure is a red data row, not a harness crash, so still exit 0.
function fatalSetup {
  echo "Error: ${1}"
  rockeResult setup "${2}" 1 "${1}"
  exit 0
}

# Directory locks avoid inheritable file descriptors, so an external tool (or a
# daemon it starts) cannot retain a nightly lock after this worker exits.
HeldLockDirs=()
# Populate the shim named in PATH above. Separate from the PATH line because it
# needs fatalSetup, and definitions in this file come after that line runs.
function installCodShim {
  if ! mkdir -p "${CodShim}" \
    || ! ln -sf "${AOMP}/bin/clang++" "${CodShim}/c++" \
    || ! ln -sf "${AOMP}/bin/clang" "${CodShim}/cc"; then
    fatalSetup "cannot create the COD compiler shim in ${CodShim}" harness
  fi
}

# shellcheck disable=SC2317 # invoked indirectly by the EXIT trap
function cleanupOnExit {
  local Lock Owner
  for Lock in "${HeldLockDirs[@]}"; do
    Owner="${Lock}/owner"
    if [[ -f "${Owner}" && "$(< "${Owner}")" == "${BASHPID}" ]]; then
      rm -f "${Owner}"
      rmdir "${Lock}" 2>/dev/null || true
    fi
  done
  [[ -n "${RowLog}" ]] && rm -f "${RowLog}"
  [[ -n "${CodShim}" ]] && rm -rf "${CodShim}"
  return 0
}
trap cleanupOnExit EXIT

function acquireDirLock {  # <lock-dir> <description>
  local Lock="${1}" Description="${2}" OwnerPid Entry Stale Waited=0 Announced=0
  while ! mkdir "${Lock}" 2>/dev/null; do
    [[ ! -L "${Lock}" ]] \
      || fatalSetup "refusing symlinked ${Description} lock: ${Lock}" lock
    (( Waited < ROCKE_LOCK_WAIT )) || fatalSetup \
      "gave up after ${ROCKE_LOCK_WAIT}s waiting for the ${Description} lock: ${Lock}" \
      lock
    OwnerPid="$(cat "${Lock}/owner" 2>/dev/null || true)"
    # /proc, not kill -0: kill reports EPERM for a live process owned by another
    # user, and treating that as dead would quarantine a held lock and let two
    # runs into the same tree.
    if [[ "${OwnerPid}" =~ ^[0-9]+$ && -d "/proc/${OwnerPid}" ]]; then
      # Say it once: an unexplained silent wait looks like a hang.
      (( Announced )) || echo "waiting for the ${Description} lock held by pid ${OwnerPid}: ${Lock}"
      Announced=1
      sleep 1; (( ++Waited ))
      continue
    fi
    if [[ ! "${OwnerPid}" =~ ^[0-9]+$ ]]; then
      sleep 1; (( ++Waited ))
      [[ ! -e "${Lock}" ]] && continue
      OwnerPid="$(cat "${Lock}/owner" 2>/dev/null || true)"
      [[ "${OwnerPid}" =~ ^[0-9]+$ ]] || fatalSetup \
        "malformed ${Description} lock owner: ${Lock}" lock
      continue
    fi
    Entry="$(find "${Lock}" -mindepth 1 -maxdepth 1 ! -name owner -print -quit 2>/dev/null)"
    if [[ -d "${Lock}" && -z "${Entry}" ]]; then
      Stale="${Lock}.stale.${BASHPID}"
      [[ ! -e "${Stale}" ]] \
        || fatalSetup "stale-lock quarantine already exists: ${Stale}" lock
      if mv "${Lock}" "${Stale}" 2>/dev/null; then
        rm -f "${Stale}/owner"
        rmdir "${Stale}" 2>/dev/null \
          || fatalSetup "cannot remove stale ${Description} lock: ${Stale}" lock
        continue
      fi
      # Someone else got there first, or the directory is not ours to move:
      # retry on the same terms as a live owner rather than spinning on the CPU.
      sleep 1; (( ++Waited ))
      continue
    fi
    fatalSetup "cannot recover stale ${Description} lock: ${Lock}" lock
  done
  printf '%s\n' "${BASHPID}" > "${Lock}/owner" \
    || fatalSetup "cannot record ${Description} lock owner: ${Lock}" lock
  HeldLockDirs+=("${Lock}")
}

# A directory that exists is not necessarily one pytest can build a collector
# for; an argument it cannot collect from aborts the whole run. Require visible
# test files, matching pytest's default python_files patterns.
function hasTests {  # <dir>
  [[ -d "${1}" && -r "${1}" ]] || return 1
  [[ -n "$(find "${1}" \( -name 'test_*.py' -o -name '*_test.py' \) \
             -print -quit 2>/dev/null)" ]]
}

# pytest reports a usage error (an unopenable root, conftest or plugin) on its
# output only, so a row saying just "exited with status 4" is undiagnosable.
function runnerDetail {  # <runner-log>
  local Log="${1:-}" Line
  [[ -n "${Log}" && -f "${Log}" ]] || return 0
  Line="$(grep -m1 -E '^(ERROR|ImportError while loading)' "${Log}")" || return 0
  [[ -n "${Line}" ]] && printf ': %s' "${Line:0:160}"
}

# Build rocKE's C++ engine extension (`rocke_engine`) so the pytest lanes can
# import it, reporting where it landed in EngineExtDir.
#
# rocKE's cross-engine tests skip without it, and it is not a side concern: the
# extension is 200k lines of C++ compiled by the COD, and the tests it unlocks
# compare the COD-built engine against the Python one. Built through rocKE's own
# ROCKE_BUILD_PYBIND option -- one tree yields both the archive and the module -- so
# there is no second recipe of ours to keep in step with theirs.
#
# It is shared across lanes rather than per-lane, so the 'all' run builds it once.
# That puts it outside the per-lane build dir ROCKE_REBUILD cleans, so this honours
# that knob itself: a nightly gets a build from scratch (a stale CMake cache survives
# a source move, which is exactly the drift a fast-moving upstream produces), while a
# hand rerun reuses whatever is still newer than rocKE's C++ sources. The stamp is the
# run's identity, so only the first lane of an 'all' run pays for the rebuild.
#
# It reports through a global because it also prints progress and, on failure, a
# result row: a caller capturing stdout would swallow the row into a variable.
EngineExtDir=""
function ensureEngineExtension {  # sets EngineExtDir
  local Root="${ROCKE_CI_BUILD_ROOT}/engine-ext" Ext Stamp Run
  EngineExtDir=""
  Stamp="${Root}/.rocke-run"
  Run="${ROCKE_RUN_ID}"
  if [[ ! -d "${ROCKE_TOP}/cpp" ]]; then
    rockeResult setup engine-extension 1 \
      "no ${ROCKE_TOP}/cpp: cannot tell whether the engine extension is current" \
      "${LaneRelevance}"
    return 1
  fi
  # Lock before deciding, not after: the decision reads the tree and one arm of it
  # deletes the tree, so an unlocked decision can race a concurrent build into
  # either a half-linked module or an rm -rf underneath it. prepareBuildRoot orders
  # it this way for the same reason.
  mkdir -p "${Root}"
  acquireDirLock "${Root}.lock" "engine extension build"
  if [[ "${ROCKE_REBUILD}" == 1 && "$(cat "${Stamp}" 2>/dev/null)" != "${Run}" ]]; then
    rm -rf "${Root}"
    mkdir -p "${Root}"
  else
    Ext="$(find "${Root}" -name 'rocke_engine*.so' -print -quit 2>/dev/null)"
    # Ask directly whether any C++ source is newer than the module we already have,
    # rather than sorting the tree: one stat walk, and no filename can confuse it.
    if [[ -n "${Ext}" ]] && [[ -z "$(find "${ROCKE_TOP}/cpp" -newer "${Ext}" \
         \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name 'CMakeLists.txt' \) \
         -print -quit 2>/dev/null)" ]]; then
      EngineExtDir="$(dirname "${Ext}")"; return 0
    fi
  fi
  local PyBind
  PyBind="$("${PyBin}" -m pybind11 --cmakedir 2>/dev/null)"
  if [[ -z "${PyBind}" ]]; then
    rockeResult setup engine-extension 1 \
      "pybind11 unavailable in ${ROCKE_VENV}; rocKE's cross-engine tests cannot run" \
      "${LaneRelevance}"
    return 1
  fi
  echo "building the rocKE C++ engine extension (${Root})"
  if ! cmake -S "${ROCKE_TOP}" -B "${Root}" -DCMAKE_BUILD_TYPE=Release \
         -DROCKE_BUILD_PYBIND=ON -Dpybind11_DIR="${PyBind}" \
         -DPython3_EXECUTABLE="${PyBin}" > "${Root}/configure.log" 2>&1; then
    rockeResult setup engine-extension 1 \
      "cmake configure failed for the engine extension (see ${Root}/configure.log)" \
      "${LaneRelevance}"
    return 1
  fi
  if ! cmake --build "${Root}" --target rocke_core rocke_engine \
         -j"$(nproc 2>/dev/null || echo 4)" > "${Root}/build.log" 2>&1; then
    # The COD compiles this, so a failure here is a genuine COD finding, not noise.
    rockeResult setup engine-extension 1 \
      "COD build of the engine extension failed (see ${Root}/build.log)" \
      "${LaneRelevance}"
    return 1
  fi
  Ext="$(find "${Root}" -name 'rocke_engine*.so' -print -quit 2>/dev/null)"
  if [[ -z "${Ext}" ]]; then
    rockeResult setup engine-extension 1 \
      "engine extension not produced under ${Root}" "${LaneRelevance}"
    return 1
  fi
  # Stamped only now: a failed build must not claim this run already rebuilt.
  printf '%s\n' "${Run}" > "${Stamp}"
  EngineExtDir="$(dirname "${Ext}")"
}

# Run pytest from the tests dir under the relevance plugin, teeing to <log> so a
# usage error stays diagnosable. Returns pytest's own status.
function runPytest {  # <xml> <manifest> <log> [pytest args...]
  local Xml="${1}" Manifest="${2}" Log="${3}"; shift 3
  # A reused build dir (ROCKE_REBUILD=0) still holds the previous report; if
  # this run never writes one, emitJunit would republish it as today's verdict.
  rm -f "${Xml}" "${Manifest}" "${Log}"
  # --continue-on-collection-errors, because upstream moves files weekly: without it
  # one unimportable module aborts collection and the lane reports two rows instead
  # of a thousand, which reads as a catastrophic night when five rocKE tests broke.
  ( cd "${ROCKE_TOP}/tests" \
    && ROCKE_RELEVANCE_OUT="${Manifest}" \
       PYTHONPATH="${PYTHONPATH}:${HelperDir}${EngineExtDir:+:${EngineExtDir}}" \
       "${PyBin}" -m pytest "$@" -p rocke_relevance -q \
         --continue-on-collection-errors --junitxml="${Xml}" ) 2>&1 \
    | tee "${Log}"
  return "${PIPESTATUS[0]}"
}

# Turn a JUnit report into result rows, or emit a red row when it is missing.
# Preserve an unexplained nonzero runner exit even when it left a partial XML
# containing only completed, passing testcases.
function emitJunit {  # <xml> <default-group> [runner-status] [manifest] [runner-log]
  local Xml="${1}" GroupDefault="${2}" RunStatus="${3:-0}" Manifest="${4:-}"
  local RunnerLog="${5:-}"
  local ParseStatus=0
  # The status a runner uses for "tests ran, some failed", which is not an error of
  # ours. Set by the caller (ctest uses 8), because deriving it from the group label
  # meant renaming a group silently changed how the runner's status was read.
  local ExpectedFailureStatus="${ExpectedRunnerFailure:-1}"
  local -a RelevanceArgs=(--relevance-default "${LaneRelevance}")
  # Pass the path whenever the lane asked for one, present or not: gating on -f
  # here meant a plugin that failed to write it produced no --relevance, hence no
  # red row, and a thousand rows quietly fell back to the lane default. The
  # converter reports an unreadable manifest itself.
  [[ -n "${Manifest}" ]] && RelevanceArgs+=(--relevance "${Manifest}")
  # A lane whose every test drives the toolchain by construction says so, so a row
  # cannot be reported below that even when the evidence is empty (work in a child).
  [[ -n "${LaneRelevanceFloor:-}" ]] \
    && RelevanceArgs+=(--relevance-floor "${LaneRelevanceFloor}")
  if [[ -f "${Xml}" ]]; then
    "${PyBin}" "${HelperDir}/rocke_junit_results.py" \
      --junit "${Xml}" --group-default "${GroupDefault}" \
      "${RelevanceArgs[@]}" || ParseStatus=$?
    if (( ParseStatus != 0 )); then
      rockeResult setup "${GroupDefault}-junit" 1 \
        "cannot parse JUnit report (status ${ParseStatus})"
    fi
    if (( RunStatus != 0 )) \
      && { (( RunStatus != ExpectedFailureStatus )) \
        || ! grep -Eq '<(failure|error)[ />]' "${Xml}"; }; then
      rockeResult setup "${GroupDefault}-runner" 1 \
        "test runner exited with status ${RunStatus}$(runnerDetail "${RunnerLog}")"
    fi
  else
    rockeResult setup "${GroupDefault}-report" 1 \
      "no JUnit report produced$(runnerDetail "${RunnerLog}")"
  fi
}

# Reuse an existing venv, else create one (numpy + pytest) outside the source
# tree; fall back to the system python only if the venv cannot be built.
function setupPython {
  local Need='import numpy, pytest' Existed=1
  PyBin=""
  if [[ -x "${ROCKE_VENV}/bin/python" ]]; then
    PyBin="${ROCKE_VENV}/bin/python"
    # An interrupted or offline bootstrap leaves a venv that every later run
    # would adopt and then fail on; top it up instead of inheriting the damage,
    # unless the caller asked this script to install nothing.
    if [[ "${ROCKE_SETUP_VENV}" == "1" ]] && ! "${PyBin}" -c "${Need}" 2>/dev/null; then
      "${PyBin}" -m pip install --quiet numpy pytest || true
    fi
  elif [[ "${ROCKE_SETUP_VENV}" == "1" ]]; then
    [[ -e "${ROCKE_VENV}" ]] || Existed=0
    if python3 -m venv "${ROCKE_VENV}" \
      && "${ROCKE_VENV}/bin/python" -m pip install --quiet --upgrade pip \
      && "${ROCKE_VENV}/bin/python" -m pip install --quiet numpy pytest; then
      PyBin="${ROCKE_VENV}/bin/python"
    elif (( Existed == 0 )); then
      rm -rf "${ROCKE_VENV}"  # only what this run created
    fi
  fi
  if [[ -z "${PyBin}" ]]; then
    echo "# WARN: venv unavailable, falling back to system python3"
    # Name the real problem here: an empty PyBin would fail the import check
    # below and report a missing module instead of a missing interpreter.
    PyBin="$(command -v python3)" \
      || fatalSetup "venv unavailable and no python3 in PATH" python
  fi
  # Print once: the 'all' children resolve the same PyBin and would only repeat it.
  [[ "${InternalAllChild:-0}" == 1 ]] || echo "# PyBin=${PyBin}"
  # pytest matters as much as numpy: without it a lane exits 1 with no report
  # and the row blames a missing JUnit file.
  "${PyBin}" -c "${Need}" 2>/dev/null \
    || fatalSetup "numpy and pytest must be importable with ${PyBin} (venv ${ROCKE_VENV})" python
}

# Refuse to install into an interpreter this script did not create. Standalone,
# PyBin can be the engineer's system python3, and these are large packages that
# would stay behind in ~/.local long after the run.
function pipInstallable {  # <what>
  [[ "${PyBin}" == "${ROCKE_VENV}/bin/"* ]] && return 0
  echo "WARNING: not installing ${1} into ${PyBin}: outside ${ROCKE_VENV}." \
       "Allow the venv (ROCKE_SETUP_VENV=1) or preinstall it yourself."
  return 1
}

# Dependencies declared by rocKE's platform `[project.optional-dependencies].dev`
# that are needed by its package-local heuristics tests. Keep torch separate: it
# must match the GPU stack and is provisioned only by gpu-numeric.
# rocKE's own declared dev extras, so a dependency it adds arrives here without an
# edit. A hardcoded list silently rots: rocKE has declared pybind11 for a while and
# our list omitted it, which left every cross-engine test unable to run.
# ${ROCKE_EXTRA_TEST_DEPS} covers what rocKE uses but does not declare.
function rockeDeclaredTestDeps {
  # Diagnostics go to stderr on purpose: the caller captures stdout, and a parse
  # failure has to say *why* in the log rather than look like an empty extras list.
  "${PyBin}" - "${ROCKE_TOP}/pyproject.toml" <<'PY'
import re, sys

path = sys.argv[1]
try:
    text = open(path, encoding="utf-8").read()
except OSError as exc:
    sys.exit(f"cannot read {path}: {exc}")

specs = []
try:  # an exact parse wherever the interpreter has it (3.11+)
    import tomllib

    specs = (
        tomllib.loads(text)
        .get("project", {})
        .get("optional-dependencies", {})
        .get("dev", [])
    )
except ModuleNotFoundError:  # 3.10: read the one table we need
    # Closing bracket at line start, not the first one seen: a spec carrying its own
    # extras ("pandas[performance]") ends the list early otherwise, and installing a
    # silently short list is the rot this reads their file to avoid.
    # Anchored inside the table we mean: a bare `dev = [` search would happily take
    # a PEP 735 [dependency-groups] list, which is a different set of requirements.
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

# Whole requirement, version bound included. Reducing these to bare names let an
# installed pybind11 2.x satisfy rocKE's `pybind11>=3.0`, so pip did nothing and the
# bound rocKE had just raised was never applied -- the rot this reads their file to
# avoid. One per line, because a PEP 508 spec may contain spaces.
specs = [s.strip() for s in dict.fromkeys(specs) if s.strip()]
if not specs:
    sys.exit(f"no [project.optional-dependencies].dev entries in {path}")
print("\n".join(specs))
PY
}

function ensureProjectTestDeps {
  local -a Deps=() Extra=()
  # Read on its own, not folded in with ROCKE_EXTRA_TEST_DEPS: a non-empty extra
  # would make the combined list look healthy and hide the very rot this reads
  # rocKE's file to avoid -- installing a stale subset and leaving its own tests
  # unable to run.
  mapfile -t Deps < <(rockeDeclaredTestDeps)
  if (( ${#Deps[@]} == 0 )); then
    rockeResult setup python-test-deps 1 \
      "cannot read rocKE's declared dev dependencies from ${ROCKE_TOP}/pyproject.toml"
    return 1
  fi
  read -ra Extra <<< "${ROCKE_EXTRA_TEST_DEPS}"
  Deps+=("${Extra[@]}")
  # Import names differ from distribution names often enough that checking them is
  # its own maintenance burden; pip already decides in milliseconds when satisfied.
  echo "provisioning rocKE-declared test dependencies: ${Deps[*]}"
  if pipInstallable "rocKE dev test dependencies" \
    && "${PyBin}" -m pip install --quiet "${Deps[@]}"
  then
    return 0
  fi
  rockeResult setup python-test-deps 1 "cannot provision rocKE dev test dependencies"
  return 1
}

# COD ROCm major.minor, from this install's own metadata only: rocKE's comgr
# resolver falls back to /opt/rocm when COD metadata is absent, and that
# unrelated version would select an incompatible multi-gigabyte torch wheel.
function codRocmVersion {
  head -1 "${RocmRoot}/.info/version" 2>/dev/null | cut -d. -f1,2
}

# Datalayout generation a ROCm major.minor implies (>= 7.2 emits the indexed p8 shape).
# Empty when the version is unusable.
function rocmEra {  # <major[.minor[.patch]]>
  local Major="${1%%.*}" Rest="${1#*.}" Minor=0
  # A bare major must not borrow itself as the minor: "7" is 7.0, not 7.7.
  [[ "${1}" == *.* ]] && Minor="${Rest%%.*}"
  [[ "${Major}" =~ ^[0-9]+$ && "${Minor}" =~ ^[0-9]+$ ]] || return 1
  # The datalayout generation, not the flavor. What a torch build has to match is
  # the IR shape its HIP runtime speaks, and every flavor from llvm22 on shares the
  # indexed p8 shape -- so this stays correct when rocKE adds a flavor, which a copy
  # of its release ladder did not (it kept saying llvm22 after llvm23 arrived).
  if (( Major > 7 || (Major == 7 && Minor >= 2) )); then echo indexed-p8; else echo plain-p8; fi
}

# Accept a torch that can serve as the numeric reference for this COD.
#
# Not an exact ROCm match: a COD carries an in-development ROCm (7.15, 10.0) that
# no published torch will ever match, so requiring equality makes the lane dead on
# every COD. What keeps the toolchain honest is ROCKE_COMGR_LIB, which pins rocKE
# to the COD comgr ahead of any torch-bundled one; torch's job here is to be a
# numeric oracle. So require only the same IR flavor era -- across eras its HIP
# runtime pairs badly with COD kernels -- and note a difference within one.
function validateTorch {
  local Ver TorchVer TorchMajorMinor CodEra TorchEra
  "${PyBin}" -c 'import torch' 2>/dev/null || return 1
  TorchVer="$("${PyBin}" -c 'import torch; print(torch.version.hip or "")' 2>/dev/null)"
  [[ -n "${TorchVer}" ]] || {
    echo "ERROR: ROCKE_VENV contains a non-ROCm torch build"
    return 1
  }
  TorchMajorMinor="$(cut -d. -f1,2 <<< "${TorchVer}")"
  Ver="$(codRocmVersion)"
  CodEra="$(rocmEra "${Ver}" || true)"
  TorchEra="$(rocmEra "${TorchMajorMinor}" || true)"
  if [[ -n "${CodEra}" && -n "${TorchEra}" && "${CodEra}" != "${TorchEra}" ]]; then
    echo "ERROR: torch ROCm ${TorchVer} speaks the ${TorchEra} datalayout, COD ROCm ${Ver} the ${CodEra} one"
    return 1
  fi
  if [[ -n "${Ver}" && "${TorchMajorMinor}" != "${Ver}" ]]; then
    echo "using torch ROCm ${TorchVer} as the numeric reference for COD ROCm ${Ver}" \
         "(same ${CodEra} datalayout; rocKE still compiles through the COD comgr)"
  else
    echo "using torch ROCm ${TorchVer} from ${ROCKE_VENV}"
  fi
  return 0
}

# Provision rocKE's numeric reference (torch) on demand from the pytorch ROCm
# wheel index for the COD's ROCm major.minor; ROCKE_TORCH_INDEX_URL overrides it.
function ensureTorch {
  local Idx="${ROCKE_TORCH_INDEX_URL}" Ver
  validateTorch && return 0
  if [[ -z "${Idx}" ]]; then
    Ver="$(codRocmVersion)"
    if [[ -z "${Ver}" ]]; then
      echo "ERROR: COD has no .info/version; set ROCKE_TORCH_INDEX_URL or preinstall torch in ROCKE_VENV"
      return 1
    fi
    # A guess, and often a wrong one: pytorch publishes an index per *released*
    # ROCm, and a COD carries an unreleased one. The failure path below says what
    # to do about it.
    Idx="https://download.pytorch.org/whl/rocm${Ver}"
  fi
  pipInstallable "torch (multiple GB)" || return 1
  # A wheel for the wrong ROCm still satisfies the requirement, so a plain
  # install would be a no-op here and validateTorch would fail again.
  local -a Force=()
  "${PyBin}" -c 'import torch' 2>/dev/null && Force=(--force-reinstall)
  echo "provisioning torch for gpu-numeric from ${Idx}"
  "${PyBin}" -m pip install "${Force[@]}" --index-url "${Idx}" torch || {
    echo "ERROR: no torch at ${Idx}"
    echo "       An unreleased COD ROCm has no published torch. Set" \
         "ROCKE_TORCH_INDEX_URL to a released"
    echo "       index of the same era ($(rocmEra "$(codRocmVersion)" || echo '?'))," \
         "e.g. https://download.pytorch.org/whl/rocm7.2,"
    echo "       or preinstall torch in ${ROCKE_VENV}."
    return 1
  }
  validateTorch
}

# branch@shortsha of the rocKE checkout, or '?' when it is not a git tree.
function rockeSrcRev {
  local Branch Sha
  Branch="$(git -C "${ROCKE_TOP}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  Sha="$(git -C "${ROCKE_TOP}" rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "${Branch}@${Sha}"
}

# Ensure the shared rocm-libraries checkout is usable before testing. rocKE and CK
# share this tree (${AOMP_REPOS_TEST}/composable-kernels/rocm-libraries), so a
# refresh must never discard another user's work: clone only into a missing/empty
# path, require an existing checkout to be clean, fetch the requested branch, and
# advance its local branch by fast-forward only.
function updateRockeSource {
  local Top Origin SourceLock
  local Repo="${ROCKE_REPO_ROOT}"
  local Url="${ROCKE_REPO_URL}"
  local Branch="${ROCKE_REPO_BRANCH}"
  if [[ "${Repo}" != "${ROCKE_TOP}" ]]; then
    mkdir -p "$(dirname "${Repo}")" \
      || fatalSetup "cannot create source parent: $(dirname "${Repo}")" source
    SourceLock="${Repo}.rocke-ci.lock.d"
    acquireDirLock "${SourceLock}" "rocKE source"
  fi
  if [[ "${Repo}" == "${ROCKE_TOP}" ]]; then
    echo "WARN: cannot derive the rocm-libraries root from a custom ROCKE_TOP"
  elif ! git -C "${Repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [[ -d "${Repo}" && -n "$(find "${Repo}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      fatalSetup "refusing to replace non-empty non-git source path: ${Repo}" source
    fi
    echo "no rocKE checkout under ${ROCKE_TOP}; cloning rocm-libraries (several GB,"
    echo "shared with CK) into ${Repo}. Point ROCKE_TOP at an existing checkout to skip."
    rmdir "${Repo}" 2>/dev/null || true
    git clone --single-branch --depth 1 -b "${Branch}" "${Url}" "${Repo}" \
      || fatalSetup "rocm-libraries clone failed: ${Url} (${Branch})" source
  else
    Top="$(realpath -m "$(git -C "${Repo}" rev-parse --show-toplevel)")"
    [[ "${Top}" == "$(realpath -m "${Repo}")" ]] \
      || fatalSetup \
        "source path is nested in another repository (${Top}); refusing to update: ${Repo}" \
        source
    if [[ "${ROCKE_UPDATE_REPO}" == 1 ]]; then
      git check-ref-format --branch "${Branch}" >/dev/null 2>&1 \
        || fatalSetup "invalid ROCKE_REPO_BRANCH: ${Branch}" source
      Origin="$(git -C "${Repo}" remote get-url origin 2>/dev/null || true)"
      [[ "${Origin}" == "${Url}" ]] \
        || fatalSetup \
          "source origin mismatch: expected ${Url}, found ${Origin:-<none>}" \
          source
      if [[ -n "$(git -C "${Repo}" status --porcelain)" ]]; then
        fatalSetup \
          "rocm-libraries checkout has local changes; refusing to update: ${Repo}" \
          source
      fi
      echo "updating rocm-libraries (${Repo})"
      git -C "${Repo}" fetch --prune origin \
        "+refs/heads/${Branch}:refs/remotes/origin/${Branch}" \
        || fatalSetup "failed to fetch origin/${Branch}; refusing to test stale source" source
      if git -C "${Repo}" show-ref --verify --quiet "refs/heads/${Branch}"; then
        git -C "${Repo}" merge-base --is-ancestor "${Branch}" "origin/${Branch}" \
          || fatalSetup \
            "local ${Branch} is not a fast-forward of origin/${Branch}; refusing to rewrite it" \
            source
        git -C "${Repo}" switch "${Branch}" \
          || fatalSetup "failed to switch to source branch ${Branch}" source
      else
        git -C "${Repo}" switch --track -c "${Branch}" "origin/${Branch}" \
          || fatalSetup "failed to create tracking branch ${Branch}" source
      fi
      git -C "${Repo}" merge --ff-only "origin/${Branch}" \
        || fatalSetup "failed to fast-forward ${Branch}; refusing to test stale source" source
    fi
  fi
  echo "rocKE src = $(rockeSrcRev)  (rocm-libraries: ${ROCKE_TOP})"
}

function printBanner {
  local ClangVer LlvmSha HipVer
  ClangVer="$("${CXX}" --version 2>/dev/null | head -1)"
  # The COD clang embeds its llvm-project git SHA in --version; grab it so a stale
  # COD (or a re-tagged same-SHA build) is identifiable from the log alone.
  LlvmSha="$("${CXX}" --version 2>/dev/null | grep -oE '[0-9a-f]{12,40}' | tail -1)"
  HipVer="$(awk -F= '
    /^HIP_VERSION_(MAJOR|MINOR|PATCH|GITHASH)=/ { v[$1] = $2 }
    END { if (v["HIP_VERSION_MAJOR"] != "")
            printf "%s.%s.%s-%s", v["HIP_VERSION_MAJOR"], v["HIP_VERSION_MINOR"], \
                                  v["HIP_VERSION_PATCH"], v["HIP_VERSION_GITHASH"] }
  ' "${RocmRoot}/share/hip/version" 2>/dev/null)"
  echo "==============================================================================="
  echo "rocKE ${Stage}  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "  AOMP        = ${AompInput} -> ${AOMP}"
  echo "  ROCM_PATH   = ${ROCM_PATH} (${RocmRootSource})"
  echo "  clang       = ${ClangVer}"
  echo "  llvm SHA    = ${LlvmSha:-?}"
  echo "  HIP         = ${HipVer:-?}"
  echo "  flavors     = codegen:${ROCKE_CODEGEN_FLAVOR}  comgr:${ROCKE_COMGR_FLAVOR}  engine:${ROCKE_ENGINE_FLAVORS}"
  echo "==============================================================================="
}

# True when a resolved path lives inside the COD install root.
function underCod {
  [[ -n "${1}" && -e "${1}" && "$(realpath -m "${1}")" == "${RocmRoot}"/* ]]
}

# Print one hygiene row; return non-zero when a *hard* requirement is external.
function codToolchainRow {  # <label> <path> <hard:1|0>
  local Label="${1}" Path="${2}" Hard="${3}" Tag="MISSING"
  if [[ -n "${Path}" && -e "${Path}" ]]; then
    Tag="EXTERNAL"
    underCod "${Path}" && Tag="COD"
  fi
  printf '  %-13s [%-8s] %s\n' "${Label}" "${Tag}" "${Path:-<not found>}"
  [[ "${Hard}" == 1 && "${Tag}" != COD ]] && return 1
  return 0
}

# Datalayout generation the COD clang itself emits, from its target datalayout p8
# field. This cannot name a flavor: rocKE's llvm22 and llvm23 share the indexed p8
# shape, so the field distinguishes generations, not releases. It is still the one
# signal that cannot leak from an unrelated tree, which makes it the right
# cross-check against the ROCm number the comgr reports.
function codClangP8Shape {
  local Arch="${1}" Dl
  Dl=$(printf 'int _rocke_flavor_probe;\n' | "${AOMP}/bin/clang" -x c \
        -target amdgcn-amd-amdhsa -mcpu="${Arch}" -emit-llvm -S - -o - 2>/dev/null \
        | sed -n 's/^target datalayout = "\(.*\)"/\1/p')
  case "${Dl}" in
    *p8:128:128:128:48*) echo indexed ;;
    *p8:128:128-*)       echo plain ;;
    *)                   echo unknown ;;
  esac
}


# Pin one flavor knob to the COD clang's own flavor when it is 'auto'; warn when
# an explicit value disagrees, since rocKE would then lower IR in the wrong one.
function resolveFlavorKnob {  # <env-var-name> <flavor>
  local Name="${1}" Flavor="${2}" Cur="${!1}"
  if [[ "${Cur}" == auto ]]; then
    [[ "${Flavor}" != "?" && -n "${Flavor}" ]] \
      || fatalSetup "cannot determine the IR flavor for ${Name}" toolchain
    export "${Name}=${Flavor}"
  elif [[ "${Flavor}" != "?" && "${Cur}" != "${Flavor}" ]]; then
    echo "WARNING: ${Name}=${Cur} overrides the flavor this COD implies (${Flavor})"
  fi
}

# Prove the compiler-critical tools/libs resolve *inside* the COD install, so a
# green row can never come from a stale system ROCm. clang/comgr are always hard
# requirements; HIP/hipcc/llvm-readelf are hard only in the lanes that use them.
function assertCodToolchain {
  local Probe Comgr ComgrVer ComgrFlavor ComgrIface ClangShape Pin Rc=0
  local HipHard=0 HipccHard=0 ReadelfHard=0 CxxHard=0 ComgrVersionTrusted=1
  # rocke_cod_probe.py reports the comgr lib rocKE will actually load, its ROCm
  # vintage, the IR flavor *rocKE derives* from that vintage, and the lib's own
  # interface version. The flavor comes from rocKE's own ladder, so a release that
  # adds a flavor needs no edit here.
  ClangShape="$(codClangP8Shape "${ROCKE_CI_ARCHES%% *}")"
  # The clang datalayout is the one basis for the flavor that cannot leak in from an
  # unrelated tree. Losing it means the flavor falls back to the comgr's ROCm number,
  # which is exactly the value this gate spends thirty lines distrusting -- so say so
  # in a row instead of continuing on the weaker basis in silence. A new p8 shape
  # upstream lands here, and that is worth a night's attention.
  if [[ "${ClangShape}" == unknown ]]; then
    rockeResult setup cod-datalayout 1 \
      "the COD clang emits a p8 datalayout rocKE does not describe: pinning the flavor from the comgr's ROCm number instead" \
      harness
  fi
  Probe="$("${PyBin}" "${HelperDir}/rocke_cod_probe.py" "${ClangShape}" 2>/dev/null)"
  read -r Pin ComgrFlavor ComgrVer ComgrIface Comgr <<< "${Probe}"
  resolveFlavorKnob ROCKE_CODEGEN_FLAVOR "${Pin}"
  resolveFlavorKnob ROCKE_COMGR_FLAVOR "${Pin}"
  echo "COD toolchain hygiene (compiler-critical rows must read [COD]):"
  codToolchainRow clang++ "$(command -v clang++)" 1 || Rc=1
  codToolchainRow comgr "${Comgr}" 1 || Rc=1
  if [[ "${ComgrVer}" != "?" && ! -e "${RocmRoot}/.info/version" ]] && underCod "${Comgr}"; then
    ComgrVersionTrusted=0
  fi
  # rocke_cod_smoke.py pins rocKE's vintage lookup only when it cannot be
  # believed; pinning it otherwise would also satisfy rocKE's IR-flavor guard
  # and hide a genuine comgr-vs-clang split.
  export ROCKE_COMGR_VERSION_TRUSTED="${ComgrVersionTrusted}"
  if (( ComgrVersionTrusted == 1 )); then
    echo "                       comgr interface ${ComgrIface}, rocm vintage ${ComgrVer} -> rocke flavor ${ComgrFlavor}"
  else
    echo "                       comgr interface ${ComgrIface}, rocm vintage metadata unavailable in COD"
  fi
  echo "                       cod clang emits the ${ClangShape} p8 datalayout"
  # The probe reports both bases: the flavor the comgr's ROCm number implies and the
  # one it chose from the clang's datalayout generation. They differ only when those
  # two disagree, which means one of them is not from this install -- and the clang
  # wins, because a datalayout cannot leak from an unrelated tree.
  if (( ComgrVersionTrusted == 1 )) && [[ "${ComgrFlavor}" != "?" && "${Pin}" != "${ComgrFlavor}" ]]; then
    echo "WARNING: datalayout split -- comgr rocm ${ComgrVer} implies ${ComgrFlavor}, but the COD"
    echo "         clang emits the ${ClangShape} p8 shape; pinning ${Pin} to match the clang."
  fi
  # A COD shipping no .info/version lets rocke's vintage number leak from the
  # system /opt/rocm. Only worth saying for a COD-resident comgr: an external one
  # already failed hard above.
  if (( ComgrVersionTrusted == 0 )); then
    echo "WARNING: ignoring comgr rocm vintage ${ComgrVer}: it leaked from the system /opt/rocm fallback"
    echo "         ($(cat /opt/rocm/.info/version 2>/dev/null || echo '?')); the flavor knobs keep whatever rocKE derived from it."
  fi
  # A hard requirement is a property of the lanes actually running, so 'all'
  # expands to its lane list and the lane -> tool mapping is stated once.
  local Lane Running="${Stage}"
  [[ "${Stage}" == all ]] && Running="${ROCKE_ALL_LANES}"
  # shellcheck disable=SC2086 # intended word splitting of the lane list
  for Lane in ${Running}; do
    case "${Lane}" in
      pytest|gpu-numeric) HipHard=1; HipccHard=1 ;;
      cod-comgr)          HipHard=1 ;;
      perf)               ReadelfHard=1 ;;
      # rocKE builds the engine archive by invoking `c++`; the shim points that name
      # at the COD, and this row is what proves it rather than assuming it.
      engine)             CxxHard=1 ;;
    esac
  done
  codToolchainRow hip-runtime "${ROCKE_HIP_LIB}" "${HipHard}" || Rc=1
  codToolchainRow hipcc "$(command -v hipcc)" "${HipccHard}" || Rc=1
  codToolchainRow llvm-readelf "$(command -v llvm-readelf)" "${ReadelfHard}" || Rc=1
  codToolchainRow c++ "$(command -v c++)" "${CxxHard}" || Rc=1
  (( Rc == 0 )) || fatalSetup \
    "compiler toolchain resolves outside the COD (${RocmRoot}); refusing to test a stale system ROCm" \
    toolchain
}

# Ensure a CMake new enough for rocKE's block() (>= 3.25); emit a clear setup
# row and signal failure otherwise (the ctest/engine lanes build via cmake).
function requireCmake {
  local Ver Major Minor
  Ver="$(cmake --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)"
  if [[ -z "${Ver}" ]]; then
    rockeResult setup cmake 1 "cmake not found (need >= 3.25 for rocKE's block(); set ROCKE_CMAKE_BIN)"
    return 1
  fi
  Major="${Ver%%.*}"; Minor="${Ver#*.}"
  if (( Major < 3 || (Major == 3 && Minor < 25) )); then
    rockeResult setup cmake 1 "cmake ${Ver} too old (need >= 3.25; set ROCKE_CMAKE_BIN)"
    return 1
  fi
  return 0
}

# Flavors the byte-identity gate sweeps. 'auto' asks rocKE for its own published
# list, so a flavor it adds is swept the night it appears instead of waiting for
# someone here to notice; the fallback is the pair that predates the list.
#
# Reports through a global for the same reason ensureEngineExtension does: it may
# emit a result row, and a caller capturing stdout would read the row itself as the
# answer -- eight words of a red row swept as eight flavors.
EngineFlavorList=""
function engineFlavors {  # sets EngineFlavorList
  if [[ "${ROCKE_ENGINE_FLAVORS}" != auto ]]; then
    EngineFlavorList="${ROCKE_ENGINE_FLAVORS}"; return
  fi
  EngineFlavorList="$("${PyBin}" -c 'from rocke.core.lower_llvm import LLVM_FLAVORS
print(" ".join(LLVM_FLAVORS))' 2>/dev/null)"
  if [[ -z "${EngineFlavorList}" ]]; then
    EngineFlavorList="llvm20 llvm22"
    rockeResult setup engine-flavors 1 \
      "cannot read rocKE's LLVM_FLAVORS; sweeping ${EngineFlavorList} only" harness
  fi
}

# How much the byte-identity gate covered, and what it faulted on, so a result row
# can carry both instead of pointing at the log.
# Echoes the count, or nothing when the gate did not report one. Both are red, but
# they are different findings and the row must not blame rocKE's corpus for a change
# in how the gate prints.
function gateFamilyCount {  # <gate-log>
  grep -m1 -oP 'families=\K[0-9]+' "${1}" 2>/dev/null || true
}

function gateFailures {  # <gate-log>
  local Bad
  # Anchored on the per-family line only. The gate also prints "<TAG> <count>" and
  # "<TAG> families: a, b" with the same leading tags, and a looser pattern
  # harvested "2" and "families:" into the row instead of family names.
  Bad="$(grep -oP '^ {2}(DRIFT|RANGE_DRIFT|COMPILE_FAIL|MODE_UNSUPPORTED) {2,}\K[A-Za-z_][A-Za-z0-9_]*' \
         "${1}" 2>/dev/null | sort -u | head -5 | tr '\n' ' ')"
  printf '%s' "${Bad:-no family named by the gate; see the lane log}"
}

function stageEngine {
  requireCmake || return
  # The C++ archive is flavor-independent -- the flavor only changes run_diff's .ll
  # emission -- so every flavor gets the same --build-root and cmake's incrementality
  # covers the rebuilds after the first. The gate reconfigures each time regardless;
  # this only avoids recompiling.
  local -a Flavors
  engineFlavors
  read -ra Flavors <<< "${EngineFlavorList}"
  local Flavor
  for Flavor in "${Flavors[@]}"; do
    echo "byte-identity gate: ${Flavor}"
    local Gate="${BuildRoot}/byte-identity-${Flavor}.log"
    # Keep a copy of the gate's own output so the row can name what it faulted on;
    # PIPESTATUS because tee would otherwise report its own success as the verdict.
    ROCKE_LLVM_FLAVOR="${Flavor}" "${PyBin}" "${ROCKE_TOP}/tools/check_byte_identity.py" \
      --build-root "${BuildRoot}" 2>&1 | tee "${Gate}"
    if (( PIPESTATUS[0] == 0 )); then
      # An empty corpus passes this gate trivially, so the coverage it claims is part
      # of the verdict rather than decoration: too few families is a red row, however
      # green the comparison was.
      local Fams; Fams="$(gateFamilyCount "${Gate}")"
      if [[ -z "${Fams}" ]]; then
        rockeResult byte-identity "${Flavor}" 1 \
          "gate passed but reported no family count: cannot tell what it covered" \
          "${LaneRelevance}"
      elif (( Fams < ROCKE_MIN_GATE_FAMILIES )); then
        rockeResult byte-identity "${Flavor}" 1 \
          "gate passed over only ${Fams} families (floor ${ROCKE_MIN_GATE_FAMILIES}): its corpus shrank" \
          "${LaneRelevance}"
      else
        rockeResult byte-identity "${Flavor}" 0 \
          "engine == python .ll over ${Fams} families" "${LaneRelevance}"
      fi
    else
      # Name the families that drifted: "see log" sends the reader into a
      # thousand-line file for something a one-line message can carry.
      rockeResult byte-identity "${Flavor}" 1 \
        "gate RED: $(gateFailures "${Gate}")" "${LaneRelevance}"
    fi
  done
}

function stageCtest {
  local RunRc
  requireCmake || return
  echo "cmake configure"
  cmake -S "${ROCKE_TOP}" -B "${BuildRoot}" -DCMAKE_BUILD_TYPE=Release \
    || { rockeResult ctest configure 1 "cmake configure failed" "${LaneRelevance}"; return; }
  echo "cmake build"
  cmake --build "${BuildRoot}" -j"$(nproc 2>/dev/null || echo 4)" \
    || { rockeResult ctest build 1 "cmake build failed" "${LaneRelevance}"; return; }
  echo "ctest"
  local Xml="${BuildRoot}/ctest-junit.xml"
  rm -f "${Xml}"  # never let a reused build dir's stale report stand in
  ( cd "${BuildRoot}" && ctest --output-on-failure --no-tests=ignore --output-junit "${Xml}" )
  RunRc=$?
  ExpectedRunnerFailure=8 emitJunit "${Xml}" ctest "${RunRc}"
}

function stagePytest {
  local RunRc Root
  local -a TestRoots=() Missing=()
  # A root pytest cannot collect from is a usage error that ends the lane before
  # any test runs, so the ~1000 rows vanish instead of turning red. Report the
  # gap and run the roots that are usable.
  # Every root is reported the same way when it disappears. Treating one as optional
  # is how a whole tree leaves the suite unnoticed -- and library/tests is precisely
  # the tree rocKE's own gap registry says nothing else gates.
  for Root in "${ROCKE_TOP}/tests" \
              "${ROCKE_TOP}/python/rocke/benchmark" \
              "${ROCKE_TOP}/python/rocke/heuristics/tests" \
              ; do
    if hasTests "${Root}"; then TestRoots+=("${Root}"); else Missing+=("${Root}"); fi
  done
  # library/tests belongs to the surrounding rocm-libraries checkout, so it is only
  # required when we are in one: ROCKE_TOP is documented as accepting a bare rocKE
  # tree, and demanding it there would make that mode permanently red.
  if [[ "${ROCKE_REPO_ROOT}" != "${ROCKE_TOP}" ]]; then
    Root="${ROCKE_PROJECT_ROOT}/library/tests"
    if hasTests "${Root}"; then TestRoots+=("${Root}"); else Missing+=("${Root}"); fi
  fi
  if (( ${#Missing[@]} )); then
    rockeResult setup pytest-roots 1 "unusable test roots: ${Missing[*]}"
  fi
  if (( ${#TestRoots[@]} == 0 )); then
    rockeResult setup pytest-roots 1 "no pytest roots under ${ROCKE_TOP}"
    return
  fi
  ensureProjectTestDeps || return
  # rocKE's cross-engine tests need its C++ extension; without it they skip, and a
  # skip naming the toolchain is a blocked row. Its absence is reported by the
  # builder itself, so a failure here degrades the lane instead of ending it.
  ensureEngineExtension || true
  echo "relative-path guard"
  if "${PyBin}" "${ROCKE_TOP}/tests/run_all.py" --no-gate --no-pytest \
      --build-root "${BuildRoot}/guard"; then
    rockeResult guard relative-path 0 ok "${LaneRelevance}"
  else
    rockeResult guard relative-path 1 "guard failed" "${LaneRelevance}"
  fi
  echo "pytest (project unit-test roots)"
  local Xml="${BuildRoot}/pytest-junit.xml"
  local Manifest="${BuildRoot}/pytest-relevance.json"
  local Out="${BuildRoot}/pytest-output.log"
  runPytest "${Xml}" "${Manifest}" "${Out}" "${TestRoots[@]}" \
    --ignore="${ROCKE_TOP}/tests/instances/test_rocke_numeric.py"
  RunRc=$?
  emitJunit "${Xml}" pytest "${RunRc}" "${Manifest}" "${Out}"
}

function stageGpuNumeric {
  local DeviceArch RunRc
  # Pin the device here rather than in a wrapper: this lane launches kernels, and
  # the 'all' wrapper runs it too, so a wrapper-only export made the consolidated
  # nightly and the standalone lane measure different GPUs on a multi-GPU host.
  # The probe below asks about index 0 of whatever is visible, so both agree.
  export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
  # "This host has no GPU" is a green skip; "we could not ask" is not. Collapsing
  # the two would let a rocKE rename retire the only lane that can catch a
  # miscompile without a single red row, so the probe reports which happened.
  # stdout only: the probe already reports its own failure there, and folding
  # stderr in would let one HSA warning line prepend itself to the value the arms
  # below match -- promoting noise to "this is the device".
  DeviceArch="$("${PyBin}" - <<'PY'
try:
    from rocke.runtime.hip_module import get_device_arch

    print(get_device_arch(0) or "none")
except Exception as exc:  # noqa: BLE001 - any failure here means "cannot ask"
    print(f"error: {type(exc).__name__}: {exc}".replace("\n", "; "))
PY
)"
  case "${DeviceArch}" in
    none)
      rockeResult environment device Check "no ROCm GPU agent on this host"
      return ;;
    gfx[0-9a-f]*) ;;
    *)
      # Anything that is not "none" and not an arch is a broken probe, including
      # the empty string and any unexpected chatter.
      rockeResult environment device 1 \
        "cannot query the device through rocKE: ${DeviceArch:-no output}" "${LaneRelevance}"
      return ;;
  esac
  # Past the device check this host *can* certify numerics, so a missing reference
  # is a misconfiguration of ours, not an environment this lane may shrug off: it
  # is the only lane that can catch a miscompile, and a green row saying "no
  # coverage" is how that goes unnoticed. A GPU-less host already returned above.
  if ! ensureTorch || ! "${PyBin}" -c 'import torch' 2>/dev/null; then
    rockeResult environment torch 1 \
      "no numeric reference on a ${DeviceArch} host: install a ROCm torch in ${ROCKE_VENV} or set ROCKE_TORCH_INDEX_URL"
    return
  fi
  # Every case here emits a kernel, compiles it through the COD and launches it on
  # the device -- rocKE runs each in a child process, so the in-process probe sees
  # none of it and would otherwise file the most compiler-driven rows we have as
  # 'logic'.
  local LaneRelevanceFloor=compiler
  echo "numeric certification: ${DeviceArch}"
  local Xml="${BuildRoot}/numeric-junit.xml"
  local Manifest="${BuildRoot}/numeric-relevance.json"
  local Out="${BuildRoot}/numeric-output.log"
  runPytest "${Xml}" "${Manifest}" "${Out}" \
    "${ROCKE_TOP}/tests/instances/test_rocke_numeric.py"
  RunRc=$?
  emitJunit "${Xml}" "numeric-${DeviceArch}" "${RunRc}" "${Manifest}" "${Out}"
}

function stagePerf {
  # Host-only codegen signal: per arch, compile the smoke kernel with the COD
  # comgr and read its resource footprint from the HSACO's ELF notes -- no GPU,
  # no torch. A spill on this fixed kernel is a real regression. See README.
  echo "codegen resource footprint (native rocke.benchmark.perf.occupancy)"
  codSmokeSweep occupancy
}

# Run the single-arch cod smoke for one arch. Extra args (e.g. --experimental)
# are forwarded to the driver; the aborted-row keeps the same experimental tag.
function codSmoke {  # <mode> <arch> [extra driver args...]
  local Mode="${1}" Arch="${2}"; shift 2
  local -a Args=(--mode "${Mode}" --arch "${Arch}" --flavor)
  if [[ "${Mode}" == codegen ]]; then
    Args+=("${ROCKE_CODEGEN_FLAVOR}" --clang "${AOMP}/bin/clang" --out "${BuildRoot}")
  else
    Args+=("${ROCKE_COMGR_FLAVOR}")
    [[ "${Mode}" == occupancy ]] \
      && Args+=(--readelf "${AOMP}/bin/llvm-readelf")
  fi
  # Mirror the driver's grouping so a hard abort files where its own rows would;
  # a plain universal_gemm would collide across modes. See rocke_cod_smoke.py.
  local Group="universal_gemm.${Mode}"
  [[ "${Mode}" == occupancy ]] && Group="occupancy"
  local Suffix=""; [[ "$*" == *--experimental* ]] && Suffix=" (experimental)"
  "${PyBin}" "${HelperDir}/rocke_cod_smoke.py" "${Args[@]}" "$@" \
    || rockeResult "${Group}" "${Arch}${Suffix}" 1 "${Mode} driver aborted (see log)" \
         "${LaneRelevance}"
}

# Sweep the production arches, then the experimental ones (tagged as such).
function codSmokeSweep {  # <mode>
  local Mode="${1}"
  local -a Prod Experimental
  local Arch
  read -ra Prod <<< "${ROCKE_CI_ARCHES}"
  read -ra Experimental <<< "${ROCKE_CI_ARCHES_EXPERIMENTAL}"
  for Arch in "${Prod[@]}"; do codSmoke "${Mode}" "${Arch}"; done
  for Arch in "${Experimental[@]}"; do codSmoke "${Mode}" "${Arch}" --experimental; done
}

function stageCodCodegen { codSmokeSweep codegen; }
function stageCodComgr { codSmokeSweep comgr; }

# Origin of a report row: 'rocKE' = the project's own tests/tools, 'ci-harness'
# = a probe this CI adds (cod-*/perf) or its own plumbing, whichever lane hit it.
# Mirrors rocke_extract.py's area classifier. See README.md "Test origin".
function laneOrigin {  # <lane> [group]
  case "${2:-}" in setup|environment) echo ci-harness; return ;; esac
  case "${1}" in
    engine|ctest|pytest|gpu-numeric) echo rocKE ;;
    *)                               echo ci-harness ;;
  esac
}

# How a red row from a lane should be triaged when there is no per-test evidence
# for it. The COD lanes drive the compiler by construction, so they are
# 'compiler' outright; the pytest lanes measure it per test (rocke_relevance.py)
# and fall back to 'compiler-capable' -- never 'logic' -- so an unmeasured row is
# always looked at rather than silently written off. See README.md
# "Test relevance".
function laneRelevance {  # <lane>
  case "${1}" in
    cod-codegen|cod-comgr|engine|ctest|perf) echo compiler ;;
    pytest|gpu-numeric)                      echo compiler-capable ;;
    *)                                       echo unregistered ;;
  esac
}

[[ -e "${AOMP}/bin/clang++" ]] || fatalSetup "COD compiler not found: ${AOMP}/bin/clang++" compiler
[[ -f "${HelperDir}/rocke_result.py" ]] \
  || fatalSetup "worker modules not found: ${HelperDir}" helpers

# Establish context once per invocation: banner, source refresh, hygiene gate.
# Only a direct child of this script's validated `all` process may inherit that
# verdict, so an ambient variable cannot bypass the provenance checks.
InternalAllChild=0
if [[ -n "${ROCKE_INTERNAL_PARENT_PID:-}" \
  && "${ROCKE_INTERNAL_PARENT_PID}" == "${PPID}" \
  && "$(tr '\0' ' ' < "/proc/${PPID}/cmdline" 2>/dev/null)" == *run_rocke.sh*" all"* ]]; then
  InternalAllChild=1
else
  unset ROCKE_INTERNAL_PARENT_PID
fi
if (( InternalAllChild == 0 )); then
  printBanner
  updateRockeSource
  [[ -d "${ROCKE_TOP}/python/rocke" ]] || fatalSetup "rocKE source not found: ${ROCKE_TOP}" source
  setupPython
  installCodShim
  assertCodToolchain
else
  setupPython
  installCodShim
fi

# A directly invoked lane tallies its own rows for the closing summary. The
# 'all' parent tallies each child's log instead, and a child leaves the summary
# to its parent, so neither needs a row log of its own.
if (( InternalAllChild == 0 )) && [[ "${Stage}" != all ]]; then
  RowLog="$(mktemp "${TMPDIR:-/tmp}/rocke-rows-${Stage}.XXXXXX")" || RowLog=""
  export ROCKE_ROW_LOG="${RowLog}"
fi

Names=(); Pass=(); Tot=(); Secs=(); Skip=(); Fails=()

# Rows below which a lane cannot be healthy, and a red row when it falls there.
#
# Everything else in this driver reports what it *did*; nothing noticed a lane that
# quietly stopped doing anything. rocKE relocates test trees routinely, and a root
# that keeps one file behind takes the pytest lane from ~1150 rows to 1 -- all green,
# because every row that ran passed. The house reads silence as success, so shrinking
# coverage has to be an explicit failure.
#
# These are floors, not expectations: set far below today's counts so ordinary
# upstream churn never trips them, and only a collapse does. The arch and flavor
# lanes are derived from the lists they iterate, so they need no maintenance; the two
# absolute numbers are the price of noticing a corpus disappear, and a legitimate
# shrink below them is a one-line edit here with the reason in the commit.
function laneRowFloor {  # <lane>
  local -a Arches Experimental
  read -ra Arches <<< "${ROCKE_CI_ARCHES}"
  read -ra Experimental <<< "${ROCKE_CI_ARCHES_EXPERIMENTAL}"
  local -a Flavors; read -ra Flavors <<< "${EngineFlavorList:-${ROCKE_ENGINE_FLAVORS}}"
  case "${1}" in
    pytest)                 echo 400 ;;   # ~1150 today
    ctest)                  echo 3 ;;     # 6 registered today
    gpu-numeric)            echo 5 ;;     # 7 test functions today, parametrised
    engine)                 echo "${#Flavors[@]}" ;;
    # Both sweeps, because codSmokeSweep runs the experimental arches too: counting
    # only the production ones let every experimental row vanish inside the floor.
    perf|cod-codegen|cod-comgr) echo "$(( ${#Arches[@]} + ${#Experimental[@]} ))" ;;
    # Not a benign default: reaching it means a lane was added to LaneOrder and not
    # to this table, and a floor of 1 would have hidden that with a green row.
    *)                      echo "?" ;;
  esac
}

function assertRowFloor {  # <lane> <row-log> <relevance>
  local Lane="${1}" Log="${2}" Relevance="${3}" Floor Rows
  Floor="$(laneRowFloor "${Lane}")"
  if [[ "${Floor}" == "?" ]]; then
    rockeResult setup "${Lane}-coverage" 1 \
      "lane ${Lane} has no row floor: add it to laneRowFloor" harness
    return
  fi
  # A lane that reported an environment blocker (no GPU, no numeric reference) never
  # reached its tests, so a low row count is that row's news and not a second finding.
  grep -aq '^ROCKE_RESULT|environment|' "${Log}" 2>/dev/null && return 0
  Rows="$(grep -ac '^ROCKE_RESULT|' "${Log}" 2>/dev/null || true)"
  (( ${Rows:-0} < Floor )) || return 0
  rockeResult setup "${Lane}-coverage" 1 \
    "only ${Rows:-0} result rows, below this lane's floor of ${Floor}: its corpus shrank" \
    "${Relevance}"
}

# Fold one lane's rows into the run summary, from whichever log holds them.
# Green rows that only record a skip are counted too: a lane that certified
# nothing must not read like one that certified everything.
function absorbLaneLog {  # <lane> <row-log> <elapsed-seconds>
  local Lane="${1}" Log="${2}" Elapsed="${3}" P T S Group Subtest Msg Tier
  # The status for an unmeasured row is rocke_result.STATUS_CHECK; awk cannot import
  # it, so this is the one place it is spelled out. It must be matched before the
  # arithmetic, because "Check"+0 is 0 and every unmeasured row would count as a pass.
  read -r P T S < <(awk -F'|' \
    '/^ROCKE_RESULT\|/ {t++; if ($4=="Check") s++; else if ($4+0==0) p++}
     END {print p+0, t+0, s+0}' "${Log}")
  Names+=("${Lane}"); Pass+=("${P}"); Tot+=("${T}"); Secs+=("${Elapsed}"); Skip+=("${S}")
  while IFS='|' read -r _ Group Subtest _ Msg Tier; do
    Fails+=("${Lane}|${Group}::${Subtest}|${Tier}|${Msg%%$'\x1f'*}")
  done < <(awk -F'|' '/^ROCKE_RESULT\|/ && $4+0!=0' "${Log}")
}

# Closing tally for whoever is watching the run. '#=' cannot collide with the
# ROCKE_RESULT contract extract-rocke.sh parses, so the CI is unaffected; the
# dashboard breakdown stays extract-rocke.sh's job and is not repeated here.
function printRunSummary {
  local i SumP=0 SumT=0 Note Fail FLane FRest FLoc FTier FReason
  echo "#= rocKE ${Stage} summary  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "#= origin: [rocKE] project test/tool  |  [ci-harness] a probe this CI adds (cod-*/perf; see README)"
  for (( i = 0; i < ${#Names[@]}; i++ )); do
    Note=""
    if (( Skip[i] > 0 )); then
      # The denominator is what was measurable, so these rows are not mistaken for
      # failures; a lane that is nothing but unmeasured rows certified nothing at all,
      # which is worth saying rather than printing 0 / 0.
      Note="  (${Skip[i]} not measured)"
      (( Skip[i] == Tot[i] )) && Note="  (${Skip[i]} not measured -- nothing certified)"
    fi
    printf '#=   %-12s %-12s %4d / %4d  %5ds%s\n' \
      "[$(laneOrigin "${Names[i]}")]" "${Names[i]}" "${Pass[i]}" \
      "$(( Tot[i] - Skip[i] ))" "${Secs[i]}" "${Note}"
    SumP=$(( SumP + Pass[i] )); SumT=$(( SumT + Tot[i] - Skip[i] ))
  done
  if (( ${#Names[@]} > 1 )); then
    printf '#=   %-12s %-12s %4d / %4d\n' "" TOTAL "${SumP}" "${SumT}"
  fi
  if (( ${#Fails[@]} > 0 )); then
    echo "#= failures (relevance = can this be a COD regression; see README):"
    for Fail in "${Fails[@]}"; do
      FLane="${Fail%%|*}"; FRest="${Fail#*|}"
      FLoc="${FRest%%|*}"; FRest="${FRest#*|}"
      FTier="${FRest%%|*}"; FReason="${FRest#*|}"
      # Enough to recognise the failure; the full text is in the row above and
      # in the extracted report.
      if (( ${#FReason} > 80 )); then FReason="${FReason:0:77}..."; fi
      printf '#=   %-12s %-16s %-10s %s  (%s)\n' \
        "[$(laneOrigin "${FLane}" "${FLoc%%::*}")]" "${FTier}" "${FLane}" "${FLoc}" \
        "${FReason}"
    done
  fi
}

# The 'all' meta-stage runs every lane in one session so the nightly yields one
# consolidated report/mail. Each lane is still its own child with its own build
# dir; a lane failure is logged but never stops the rest, and a tally follows.
if [[ "${Stage}" == all ]]; then
  export ROCKE_INTERNAL_PARENT_PID="${BASHPID}"
  Rc=0
  # shellcheck disable=SC2086 # intended word splitting of the lane list
  for Lane in ${ROCKE_ALL_LANES}; do
    # Guard against a nested 'all' in the (env-inherited) lane list -- each child
    # would re-enter this block and fork unbounded.
    [[ "${Lane}" == all ]] && { echo "WARN: skipping nested 'all' in ROCKE_ALL_LANES"; continue; }
    echo "########## lane: ${Lane} ##########"
    LaneLog="$(mktemp)"; LaneStart="${SECONDS}"
    # Absolute path (ScriptDir is cd-resolved) so a PATH-launched parent still
    # finds the child; tee keeps the live stream while we tally the lane's rows.
    "${ScriptDir}/run_rocke.sh" "${Lane}" 2>&1 | tee "${LaneLog}"
    LaneRc="${PIPESTATUS[0]}"
    if (( LaneRc != 0 )); then
      Rc=1
      rockeResult setup "lane-${Lane}" 1 "lane exited with status ${LaneRc}" \
        | tee -a "${LaneLog}"
    elif ! grep -aq '^ROCKE_RESULT|' "${LaneLog}"; then
      Rc=1
      rockeResult setup "lane-${Lane}" 1 "lane produced no result rows" \
        | tee -a "${LaneLog}"
    fi
    assertRowFloor "${Lane}" "${LaneLog}" "$(laneRelevance "${Lane}")" \
      | tee -a "${LaneLog}"
    absorbLaneLog "${Lane}" "${LaneLog}" "$(( SECONDS - LaneStart ))"
    rm -f "${LaneLog}"
  done
  if (( ${#Names[@]} == 0 )); then
    Msg="ROCKE_ALL_LANES selected no lanes"
    rockeResult setup lanes 1 "${Msg}"
    Names=(all); Pass=(0); Tot=(1); Secs=(0)
    Fails=("all|setup::lanes|harness|${Msg}")
    Rc=1
  fi
  printRunSummary
  exit "${Rc}"
fi

# Every lane runs under a marked, dedicated build root. Validate and mark the
# root before any recursive deletion; this prevents a typo in
# ROCKE_CI_BUILD_ROOT or a symlinked stage path from deleting unrelated data.
function prepareBuildRoot {
  local Marker="${ROCKE_CI_BUILD_ROOT}/.rocke-ci-root" BuildLock
  [[ "${BuildRoot}" == "${ROCKE_CI_BUILD_ROOT}/"* ]] \
    || fatalSetup "build path escapes ROCKE_CI_BUILD_ROOT: ${BuildRoot}" build-root
  if [[ -e "${ROCKE_CI_BUILD_ROOT}" && ! -d "${ROCKE_CI_BUILD_ROOT}" ]]; then
    fatalSetup "ROCKE_CI_BUILD_ROOT is not a directory: ${ROCKE_CI_BUILD_ROOT}" build-root
  fi
  if [[ ! -d "${ROCKE_CI_BUILD_ROOT}" ]]; then
    mkdir -p "${ROCKE_CI_BUILD_ROOT}" \
      || fatalSetup "cannot create build root: ${ROCKE_CI_BUILD_ROOT}" build-root
  fi
  if [[ ! -f "${Marker}" ]]; then
    if [[ -n "$(find "${ROCKE_CI_BUILD_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      fatalSetup \
        "unmarked non-empty ROCKE_CI_BUILD_ROOT; refusing recursive cleanup: ${ROCKE_CI_BUILD_ROOT}" \
        build-root
    fi
    printf 'rocKE CI build root\n' > "${Marker}" \
      || fatalSetup "cannot mark build root: ${ROCKE_CI_BUILD_ROOT}" build-root
  fi
  grep -qxF 'rocKE CI build root' "${Marker}" \
    || fatalSetup "invalid build-root marker: ${Marker}" build-root
  BuildLock="${ROCKE_CI_BUILD_ROOT}/.${Stage}.lock.d"
  acquireDirLock "${BuildLock}" "lane build"
  if [[ "${ROCKE_REBUILD}" == 1 ]]; then rm -rf "${BuildRoot}"; fi
  mkdir -p "${BuildRoot}" \
    || fatalSetup "cannot create lane build dir: ${BuildRoot}" build-root
}

# Every lane must resolve in every table keyed on a lane name. They are spread over
# a thousand lines, and each used to fall through to a plausible-looking default: a
# lane missing from laneRelevance was triaged as our own plumbing, one missing from
# laneRowFloor was floored at a single row. Both are silent, which is the one failure
# mode this driver is not allowed to have -- so an omission is a startup error.
function assertLaneTables {
  local Lane Gaps=""
  for Lane in "${LaneOrder[@]}"; do
    [[ "$(laneRelevance "${Lane}")" != unregistered ]] || Gaps+=" laneRelevance:${Lane}"
    [[ "$(laneRowFloor "${Lane}")" != "?" ]] || Gaps+=" laneRowFloor:${Lane}"
    [[ -n "$(laneOrigin "${Lane}")" ]] || Gaps+=" laneOrigin:${Lane}"
  done
  [[ -z "${Gaps}" ]] \
    || fatalSetup "lanes missing from a per-lane table:${Gaps}" harness
}

assertLaneTables
prepareBuildRoot

# Triage class for this lane's rows that carry no per-test evidence of their own.
LaneRelevance="$(laneRelevance "${Stage}")"

case "${Stage}" in
  engine)      stageEngine ;;
  ctest)       stageCtest ;;
  pytest)      stagePytest ;;
  gpu-numeric) stageGpuNumeric ;;
  perf)        stagePerf ;;
  cod-codegen) stageCodCodegen ;;
  cod-comgr)   stageCodComgr ;;
esac

# Before the tally reads the same log, so a floor breach is counted like any red row.
[[ -n "${RowLog}" ]] && assertRowFloor "${Stage}" "${RowLog}" "${LaneRelevance}"

echo "rocKE ${Stage} done: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ -n "${RowLog}" ]]; then
  absorbLaneLog "${Stage}" "${RowLog}" "${SECONDS}"
  printRunSummary
fi
exit 0
