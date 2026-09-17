#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocKE driver: run one lane against the compiler-of-the-day (COD), emit
# "ROCKE_RESULT|group|subtest|status|message|relevance" lines, close with a summary.
# Self-contained, so it can be run by hand; the nightly drives the same lanes.
#
# Usage: run_rocke.sh [-r] [-u] <lane>; -h lists the lanes and the common knobs.
# 'all' runs every lane in one process for one consolidated report.
#
# Worker modules: bin/aux/rocke. Extractor and documentation: the apps repo, under
# openmp-ci/rocKE.

# Source directives below resolve from this script's directory, not the caller's.
# shellcheck source-path=SCRIPTDIR
set -u

# mapfile, ${Part^} and associative-style lookups below need bash 4. Said here
# rather than left to a confusing failure three hundred lines in, because this
# script is also the one an engineer runs by hand on an unfamiliar machine.
if (( BASH_VERSINFO[0] < 4 )); then
  echo "ERROR: ${0##*/} needs bash 4 or newer; this is ${BASH_VERSION}" >&2
  exit 2
fi

ScriptDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The worker modules keep their rocke_ prefix: they go on the PYTHONPATH of
# rocKE's own test session, where a plain result.py would shadow the project's.
HelperDir="${ScriptDir}/aux/rocke"

# The bash side of the worker, sourced by name rather than discovered: a glob would
# quietly run whatever a stale file left in the directory, and the list is what says
# these four files are one program. Each module defines functions and constant
# tables only, so sourcing cannot change state or fail a run; a missing one is fatal
# here, before any lane has claimed to test anything.
for Module in rocke_toolchain.source rocke_env.source rocke_lanes.source; do
  [[ -r "${HelperDir}/${Module}" ]] || {
    echo "ERROR: ${HelperDir}/${Module} is missing; this driver is incomplete" >&2
    exit 2
  }
done
unset Module
# Sourced one literal path at a time, not in the loop above: shellcheck -x cannot
# follow a path built from a variable, so a loop would leave every module
# unchecked while the driver still reported clean.
# shellcheck source=aux/rocke/rocke_toolchain.source
. "${HelperDir}/rocke_toolchain.source"
# shellcheck source=aux/rocke/rocke_env.source
. "${HelperDir}/rocke_env.source"
# shellcheck source=aux/rocke/rocke_lanes.source
. "${HelperDir}/rocke_lanes.source"

# The lanes this driver knows, in the order 'all' runs them: the cheap host-only
# gates first so a broken COD is reported in seconds, the ~1000-row pytest lane
# next, then the on-device lane, and cod-occupancy last because a register-spill verdict is
# only worth reading once the kernels it measures are known to compile. That order,
# and everything else per-lane, comes from LaneRegistry in rocke_lanes.source, so the
# stage check, the help text, the 'all' list and the dispatch cannot disagree.
mapfile -t LaneOrder < <(laneNames)
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
  ROCKE_CI_ARCHES='...'      arch sweep for the cod-* lanes
  ROCKE_TOP=<dir>            rocKE platform checkout to test (else one is cloned)
  ROCKE_CI_BUILD_ROOT=<dir>  out-of-tree build root
  ROCKE_VENV=<dir>           the only interpreter this script may install into
  ROCKE_DEBUG=1              full Python tracebacks from the cod-* lanes
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
# Compile from cold. comgr caches compile results on disk, keyed partly on its own
# version id rather than a build hash, and every lane's compile is seconds: the cache
# buys nothing here and would make a green row depend on state no row can show.
export AMD_COMGR_CACHE="${AMD_COMGR_CACHE:-0}"
export CC="${AOMP}/bin/clang"
export CXX="${AOMP}/bin/clang++"
# rocKE builds its engine by invoking the bare name `c++`, which no COD ships, so it
# resolved to /usr/bin/c++ while the rows claimed COD provenance. The shim makes that
# name the COD's clang++; installCodShim fills it in, so PATH names it before it
# exists. Other bare names upstream invokes belong here, not in a patch to rocKE.
CodShim="${TMPDIR:-/tmp}/rocke-cod-shim-$$"
# COD llvm tools first, then the shim, then the install bin (hipcc, rocprofv3).
export PATH="${AOMP}/bin:${CodShim}:${RocmRoot}/bin:${PATH}"
# AOMP compiler runtime first (libomp/libomptarget), then the pinned ROCm
# runtime (libamdhip64/libhsa). Guard the tail: an unset var must not leave a
# trailing ':' -- an empty entry means CWD, which would breach COD isolation.
export LD_LIBRARY_PATH="${AOMP}/lib:${RocmRoot}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Where the house keeps test checkouts and supplemental tools is defined in
# bin/aomp_common_vars, and a copy of those paths here is a copy that rots. houseVar
# reads them out of it without running it in this shell; see the reasoning there.
: "${AOMP_REPOS_TEST:=$(houseVar AOMP_REPOS_TEST)}"
: "${AOMP_REPOS_TEST:=${HOME}/git/aomp-test}"
HouseSupp="$(houseVar AOMP_SUPP)"

# rocKE's CMakeLists uses block(), which needs CMake >= 3.25; prefer a modern
# local cmake when the distro one is older. Only this directory goes on PATH, never
# the directory of whatever `cmake` the house file settled on: that can be /usr/bin,
# and putting /usr/bin ahead of the COD would let a system clang answer to a name
# this driver has just proved belongs to the compiler under test.
: "${ROCKE_CMAKE_BIN:=${HouseSupp:-${HOME}/local}/cmake/bin}"
[[ -x "${ROCKE_CMAKE_BIN}/cmake" ]] && export PATH="${ROCKE_CMAKE_BIN}:${PATH}"
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
# Installing a numeric reference is host preparation, not part of a test run: the
# index a COD's own ROCm implies is almost never published, so doing it inside the
# nightly turns a prepared-host question into a red lane. Opt in for a one-off.
: "${ROCKE_PROVISION_TORCH:=0}"
# Set on the host that is supposed to certify numerics. There, a missing reference
# is a defect and reddens; anywhere else it is an unprepared host and stays a Check,
# because a row nobody can act on is the noise this suite exists without.
: "${ROCKE_NUMERIC_HOST:=0}"
# rocKE uses these without declaring them in its dev extras.
: "${ROCKE_EXTRA_TEST_DEPS:=pyarrow}"
# Run identity, inherited by the children of 'all' and persisted as the
# engine-extension rebuild stamp. Epoch seconds, not a bare PID: PIDs repeat, and a
# run that drew a previous one's would skip a demanded rebuild. Accepted only from a
# child of this run, so an ambient value cannot cancel that rebuild either.
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

# Read and written by the lane and environment modules; the driver owns the state
# those functions share, so that sourcing a module stays free of side effects.
# shellcheck disable=SC2034
EngineExtDir=""

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
}

# Shared with the lane module, for the same reason as EngineExtDir above.
# shellcheck disable=SC2034
EngineFlavorList=""

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
# Before every consumer of the registry, the hygiene gate included: it reads each
# lane's required tools, so a table that has not been proved usable would decide
# what provenance gets checked.
assertLaneTables

if (( InternalAllChild == 0 )); then
  printBanner
  updateRockeSource
  [[ -d "${ROCKE_TOP}/python/rocke" ]] || fatalSetup "rocKE source not found: ${ROCKE_TOP}" source
  setupPython
  installCodShim
  resolveCodFlavor
  assertCodToolchain
  printProvenance
  # After the block, so the opening of a log reads as one description of the run
  # before any result row appears.
  reportArchDrift
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
  echo "#= origin: [rocKE] project test/tool  |  [ci-harness] a probe this CI adds (cod-*; see README)"
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
  # comgr's own diagnostics, kept beside the build rather than discarded: an
  # in-process compile failure is otherwise opaque and needs a second run to explain.
  export AMD_COMGR_REDIRECT_LOGS="${BuildRoot}/comgr.log"
}

prepareBuildRoot

# Triage class for this lane's rows that carry no per-test evidence of their own.
LaneRelevance="$(laneRelevance "${Stage}")"

# Derived from the lane name, never stored, so a lane can only run its own body.
"$(laneHandler "${Stage}")"

# Before the tally reads the same log, so a floor breach is counted like any red row.
[[ -n "${RowLog}" ]] && assertRowFloor "${Stage}" "${RowLog}" "${LaneRelevance}"

echo "rocKE ${Stage} done: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ -n "${RowLog}" ]]; then
  absorbLaneLog "${Stage}" "${RowLog}" "${SECONDS}"
  printRunSummary
fi
exit 0
