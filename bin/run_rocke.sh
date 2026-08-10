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

# The lanes this driver knows, in the order 'all' runs them: the gates that prove
# the COD can build rocKE at all, cheapest first, so a broken compiler is
# reported in seconds rather than after a full run.
# Single source of truth: the stage check, the help text and the 'all' lane list
# all derive from it, so none of them can be updated without the others.
LaneOrder=(engine ctest)
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
  ROCKE_TOP=<dir>            rocKE platform checkout to test (else one is cloned)
  ROCKE_CI_BUILD_ROOT=<dir>  out-of-tree build root
  ROCKE_VENV=<dir>           the only interpreter this script may install into
  ROCKE_DEBUG=1              full Python tracebacks from the helper modules
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
# a test runner root its collection tree at / and scan shared parents like /work,
# aborting on the first unreadable entry there.
ROCKE_PROJECT_ROOT="$(realpath -m -s "${ROCKE_TOP}/..")"
# Path from the shared rocm-libraries repo root down to the rocKE platform dir.
ROCKE_TOP_SUFFIX="/dnn-providers/hip-kernel-provider/rocke/platform"
ROCKE_REPO_ROOT="${ROCKE_TOP%"${ROCKE_TOP_SUFFIX}"}"
: "${ROCKE_VENV:=${HOME}/.local/rocKE-venv}"
: "${ROCKE_CODEGEN_FLAVOR:=auto}"
: "${ROCKE_COMGR_FLAVOR:=auto}"
: "${ROCKE_CI_ARCHES:=gfx950 gfx942 gfx1151 gfx1201}"
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
# Override to scope a run, e.g. ROCKE_ALL_LANES='engine'.
: "${ROCKE_ALL_LANES:=${LaneOrder[*]}}"

# Run-shape knobs. The nightly wrappers pin both gates to 1; run by hand they
# default off, so a rerun neither wipes the lane build dir nor touches the shared
# checkout, and -r/-u opt in exactly as they do for CK's driver.
: "${ROCKE_REBUILD:=0}"
: "${ROCKE_UPDATE_REPO:=0}"
: "${ROCKE_SETUP_VENV:=1}"
: "${ROCKE_REPO_URL:=https://github.com/ROCm/rocm-libraries.git}"
: "${ROCKE_REPO_BRANCH:=develop}"
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





# Turn a JUnit report into result rows, or emit a red row when it is missing.
# Preserve an unexplained nonzero runner exit even when it left a partial XML
# containing only completed, passing testcases.
function emitJunit {  # <xml> <default-group> [runner-status]
  local Xml="${1}" GroupDefault="${2}" RunStatus="${3:-0}"
  local ParseStatus=0
  # The status a runner uses for "tests ran, some failed", which is not an error of
  # ours. Set by the caller (ctest uses 8), because deriving it from the group label
  # meant renaming a group silently changed how the runner's status was read.
  local ExpectedFailureStatus="${ExpectedRunnerFailure:-1}"
  local -a RelevanceArgs=(--relevance-default "${LaneRelevance}")
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
        "test runner exited with status ${RunStatus}"
    fi
  else
    rockeResult setup "${GroupDefault}-report" 1 \
      "no JUnit report produced"
  fi
}

# Reuse an existing venv, else create one (numpy) outside the source
# tree; fall back to the system python only if the venv cannot be built.
function setupPython {
  local Need='import numpy' Existed=1
  PyBin=""
  if [[ -x "${ROCKE_VENV}/bin/python" ]]; then
    PyBin="${ROCKE_VENV}/bin/python"
    # An interrupted or offline bootstrap leaves a venv that every later run
    # would adopt and then fail on; top it up instead of inheriting the damage,
    # unless the caller asked this script to install nothing.
    if [[ "${ROCKE_SETUP_VENV}" == "1" ]] && ! "${PyBin}" -c "${Need}" 2>/dev/null; then
      "${PyBin}" -m pip install --quiet numpy || true
    fi
  elif [[ "${ROCKE_SETUP_VENV}" == "1" ]]; then
    [[ -e "${ROCKE_VENV}" ]] || Existed=0
    if python3 -m venv "${ROCKE_VENV}" \
      && "${ROCKE_VENV}/bin/python" -m pip install --quiet --upgrade pip \
      && "${ROCKE_VENV}/bin/python" -m pip install --quiet numpy; then
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
  "${PyBin}" -c "${Need}" 2>/dev/null \
    || fatalSetup "numpy must be importable with ${PyBin} (venv ${ROCKE_VENV})" python
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
  # Published for the probes that need it: rocKE's own vintage lookup may be
  # pinned only when it cannot be believed, since pinning it otherwise would
  # also satisfy rocKE's IR-flavor guard and hide a comgr-vs-clang split.
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

# Origin of a report row: 'rocKE' = the project's own tests/tools, 'ci-harness'
# = a probe this CI adds or its own plumbing, whichever lane hit it.
# Mirrors rocke_extract.py's area classifier. See README.md "Test origin".
function laneOrigin {  # <lane> [group]
  case "${2:-}" in setup|environment) echo ci-harness; return ;; esac
  case "${1}" in
    engine|ctest)                    echo rocKE ;;
    *)                               echo ci-harness ;;
  esac
}

# How a red row from a lane should be triaged when there is no per-test evidence
# for it. The COD lanes drive the compiler by construction, so they are
# 'compiler' outright.
# and fall back to 'compiler-capable' -- never 'logic' -- so an unmeasured row is
# always looked at rather than silently written off. See README.md
# "Test relevance".
function laneRelevance {  # <lane>
  case "${1}" in
    engine|ctest)                            echo compiler ;;
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
# that keeps one file behind takes a lane from hundreds of rows to one -- all green,
# because every row that ran passed. The house reads silence as success, so shrinking
# coverage has to be an explicit failure.
#
# These are floors, not expectations: set far below today's counts so ordinary
# upstream churn never trips them, and only a collapse does. The arch and flavor
# lanes are derived from the lists they iterate, so they need no maintenance; the two
# absolute numbers are the price of noticing a corpus disappear, and a legitimate
# shrink below them is a one-line edit here with the reason in the commit.
function laneRowFloor {  # <lane>
  local -a Flavors; read -ra Flavors <<< "${EngineFlavorList:-${ROCKE_ENGINE_FLAVORS}}"
  case "${1}" in
    ctest)                  echo 3 ;;     # 6 registered today
    engine)                 echo "${#Flavors[@]}" ;;
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
esac

# Before the tally reads the same log, so a floor breach is counted like any red row.
[[ -n "${RowLog}" ]] && assertRowFloor "${Stage}" "${RowLog}" "${LaneRelevance}"

echo "rocKE ${Stage} done: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ -n "${RowLog}" ]]; then
  absorbLaneLog "${Stage}" "${RowLog}" "${SECONDS}"
  printRunSummary
fi
exit 0
