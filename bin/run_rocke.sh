#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocKE driver: run one lane against the compiler toolchain, emit
# "ROCKE_RESULT|group|subtest|status|message|relevance" rows, close with a
# summary. Runnable by hand; the nightly drives the same lanes. 'all' runs
# every lane in one process for one consolidated report; -h lists the lanes and
# the common knobs.
#
# Worker modules: bin/aux/rocke. Extractor and documentation: the apps repo,
# under openmp-ci/rocKE.

# Source directives resolve from this script's directory, not the caller's.
# shellcheck source-path=SCRIPTDIR
set -u

# mapfile and ${Var^} need bash 4; checked up front rather than at first use.
if (( BASH_VERSINFO[0] < 4 )); then
  echo "ERROR: ${0##*/} needs bash 4 or newer; this is ${BASH_VERSION}" >&2
  exit 2
fi

ScriptDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The worker modules keep their rocke_ prefix: they go on the PYTHONPATH of
# rocKE's own test session, where a plain result.py would shadow the project's.
HelperDir="${ScriptDir}/aux/rocke"

# The bash side of the worker, sourced by explicit name rather than by glob, so
# a stale file cannot join it. A missing module is fatal before any lane runs.
for Module in rocke_toolchain.source rocke_env.source rocke_lanes.source; do
  [[ -r "${HelperDir}/${Module}" ]] || {
    echo \
      "ERROR: ${HelperDir}/${Module} is missing; this driver is incomplete" >&2
    exit 2
  }
done
unset Module

# One literal path per source, not a loop: shellcheck -x cannot follow a path
# built from a variable, which would leave every module unchecked.
# shellcheck source=aux/rocke/rocke_toolchain.source
. "${HelperDir}/rocke_toolchain.source"
# shellcheck source=aux/rocke/rocke_env.source
. "${HelperDir}/rocke_env.source"
# shellcheck source=aux/rocke/rocke_lanes.source
. "${HelperDir}/rocke_lanes.source"

# Lane order for 'all': cheap host-only gates first, then on-device lanes,
# occupancy last, since a register-spill verdict only counts once the kernels
# compile. The registry is the single source, so stage check, help, 'all' and
# dispatch agree.
mapfile -t LaneOrder < <(listLaneNames)
Lanes="all|$(IFS="|"; printf '%s' "${LaneOrder[*]}")"

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

# --- Usage and arguments

# The help text, built from the lane registry so it cannot list a lane that is
# gone.
function printUsage {
  cat <<EOF
usage: ${0##*/} [-r] [-u] <${Lanes//|/ | }>

Runs one rocKE lane against the compiler toolchain and prints ROCKE_RESULT rows;
'all' runs every lane and adds a pass/total tally per lane. A failing test is a
result row, not a driver error, so a red row does not change the exit \
status: read
the closing summary or the rows, not \$?. ('all' does exit non-zero if a \
lane died
or produced no rows at all -- that is a broken run, not a test result.)

  -r  rebuild each lane from a clean build dir   (ROCKE_REBUILD=1; off by \
default)
  -u  refresh the shared rocm-libraries checkout (ROCKE_UPDATE_REPO=1; off by \
default)
  -h  this help

Common knobs (every ROCKE_* default is set together at the top of this script;
the lane table and the full list are in openmp-ci/rocKE/README.md):
  AOMP=<llvm dir>            the compiler toolchain to test
  ROCKE_ALL_LANES='...'      lanes 'all' runs, in order
  ROCKE_CI_ARCHES='...'      arch sweep for the smoke lanes
  ROCKE_TOP=<dir>            rocKE platform checkout to test (else one is \
cloned)
  ROCKE_CI_BUILD_ROOT=<dir>  out-of-tree build root
  ROCKE_VENV=<dir>           the only interpreter this script may install into
  ROCKE_DEBUG=1              full Python tracebacks from the smoke lanes
EOF
}

# Read the options and the lane name.
function parseArguments {  # <driver arguments>
  # -r/-u are sugar for the ROCKE_REBUILD / ROCKE_UPDATE_REPO gates, exported
  # so that the lane children of 'all' inherit the same choice.
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
  (( $# > 1 )) \
    && { echo "ERROR: unexpected argument: ${2}" >&2; printUsage; exit 2; }
  InheritedPythonPath="${PYTHONPATH:-}"

  if [[ -z "${Stage}" ]]; then
    printUsage; exit 2
  elif [[ "|${Lanes}|" != *"|${Stage}|"* ]]; then
    echo "unknown stage: ${Stage}"; printUsage; exit 2
  fi
}

# --- Configuration

# Everything derived from the compiler toolchain: its root, its ROCm
# version, and the environment every lane compiles under. Declarative.
function configureToolchain {
  # House-wide paths and defaults, including AOMP itself. The file predates
  # set -u, so it is read with the check relaxed.
  set +u
  # shellcheck disable=SC1091
  . "${ScriptDir}/aomp_common_vars"
  set -u

  # AOMP alone selects the compiler toolchain; comgr, the HIP runtime, hipcc
  # and device-libs derive from its install, ruling out a system ROCm
  # (/opt/rocm) or /usr. It may name an aomp install root or the llvm directory
  # inside one; what the lanes need is the directory whose bin/ holds clang++.
  if [[ ! -x "${AOMP}/bin/clang++" && -x "${AOMP}/llvm/bin/clang++" ]]; then
    AOMP="${AOMP}/llvm"
  fi
  AompInput="${AOMP}"
  AOMP="$(realpath -m "${AompInput}")"
  export AOMP

  # ROCM_PATH may override the derived root only while it is a realpath prefix
  # of AOMP; otherwise an ambient ROCM_PATH=/opt/rocm would pass the hygiene
  # gate.
  DerivedRoot="$(resolveRocmRoot "${AOMP}")"
  if [[ -n "${ROCM_PATH:-}" ]]; then
    EnvRoot="$(realpath -m "${ROCM_PATH}")"
    if [[ "${AOMP}/" == "${EnvRoot}/"* ]]; then
      RocmRoot="${EnvRoot}"
      RocmRootSource="from env ROCM_PATH"
    else
      echo "WARNING: ignoring ROCM_PATH=${EnvRoot} --" \
        "not a prefix of AOMP=${AOMP}"
      echo "         (looks like an ambient/system ROCm). Using the" \
        "AOMP-derived root."
      echo "         To force a specific root, point ROCM_PATH at an" \
        "ancestor of AOMP."
      RocmRoot="${DerivedRoot}"
      RocmRootSource="derived from AOMP (ROCM_PATH ignored)"
    fi
  else
    RocmRoot="${DerivedRoot}"
    RocmRootSource="derived from AOMP"
  fi

  # The ROCm version of this install, for the torch generation check and the
  # provenance block. The knob states it where the install cannot.
  : "${ROCKE_ROCM_VERSION:=}"
  resolveRocmVersion || true

  export ROCM_PATH="${RocmRoot}"
  export HIP_PATH="${RocmRoot}"

  # Pin hipcc's clang to the toolchain llvm so the HIP path cannot pick a
  # system clang.
  export HIP_CLANG_PATH="${AOMP}/bin"
  export ROCKE_COMGR_LIB="${ROCKE_COMGR_LIB:-${RocmRoot}/lib/libamd_comgr.so}"
  export ROCKE_HIP_LIB="${ROCKE_HIP_LIB:-${RocmRoot}/lib/libamdhip64.so}"

  # Compile from cold: comgr keys its on-disk cache on its own version id, not
  # a build hash, so a hit could carry results from a different compiler.
  export AMD_COMGR_CACHE="${AMD_COMGR_CACHE:-0}"
  export CC="${AOMP}/bin/clang"
  export CXX="${AOMP}/bin/clang++"

  # rocKE builds its engine via the bare name `c++`, which no toolchain ships
  # and which would otherwise resolve to /usr/bin/c++; the shim maps it to the
  # toolchain's clang++.
  ToolchainShim="${TMPDIR:-/tmp}/rocke-toolchain-shim-$$"

  # toolchain llvm tools first, then the shim, then the install bin (hipcc,
  # rocprofv3).
  export PATH="${AOMP}/bin:${ToolchainShim}:${RocmRoot}/bin:${PATH}"

  # Compiler runtime first (libomp/libomptarget), then the pinned ROCm runtime.
  # The tail guard matters: an empty entry from a trailing ':' means CWD.
  export LD_LIBRARY_PATH="${AOMP}/lib:${RocmRoot}/lib\
${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

# Where the suite reads and writes: the checkout under test, the build root,
# the virtual environment, and the cmake it needs.
function configurePaths {
  # rocKE's CMakeLists uses block(), which needs CMake >= 3.25, so prefer a
  # modern local cmake. Only this directory goes on PATH: a house cmake can
  # live in /usr/bin, and /usr/bin ahead of the toolchain would let a system
  # clang answer.
  : "${ROCKE_CMAKE_BIN:=${AOMP_SUPP}/cmake/bin}"
  [[ -x "${ROCKE_CMAKE_BIN}/cmake" ]] \
    && export PATH="${ROCKE_CMAKE_BIN}:${PATH}"

  : "${ROCKE_TOP:=${AOMP_REPOS_TEST}/composable-kernels/rocm-libraries\
/dnn-providers/hip-kernel-provider/rocke/platform}"

  # -s keeps this in ROCKE_TOP's path namespace: a symlink-resolved path can
  # land under a different prefix, making pytest root its collection tree at /
  # and scan upward.
  ROCKE_PROJECT_ROOT="$(realpath -m -s "${ROCKE_TOP}/..")"

  # Path from the shared rocm-libraries repo root down to the rocKE platform
  # dir.
  ROCKE_TOP_SUFFIX="/dnn-providers/hip-kernel-provider/rocke/platform"
  ROCKE_REPO_ROOT="${ROCKE_TOP%"${ROCKE_TOP_SUFFIX}"}"

  # Sibling of the rocm-libraries checkout, matching the ck-src/ck-build layout
  # beside it. Off /tmp, so no reaper can drop the tree or its root marker.
  if [[ "${ROCKE_REPO_ROOT}" == "${ROCKE_TOP}" \
    && -z "${ROCKE_CI_BUILD_ROOT:-}" ]]; then
    # Outside a rocm-libraries checkout the derived root lands in the user's
    # own tree.
    echo "WARNING: ROCKE_TOP is not inside a rocm-libraries checkout;" \
      "the build root"
    echo "         will be derived next to it." \
      "Set ROCKE_CI_BUILD_ROOT to choose one."
  fi
  : "${ROCKE_CI_BUILD_ROOT:=${ROCKE_REPO_ROOT%/*}/rocke-build}"
  ROCKE_CI_BUILD_ROOT="$(realpath -m "${ROCKE_CI_BUILD_ROOT}")"
}

# The suite's own knobs. Declarative; every default lives here, so one place
# shows how a run is configured.
function configureKnobs {
  : "${ROCKE_VENV:=${HOME}/.local/rocKE-venv}"
  : "${ROCKE_CODEGEN_FLAVOR:=auto}"
  : "${ROCKE_COMGR_FLAVOR:=auto}"
  : "${ROCKE_CI_ARCHES:=gfx950 gfx942 gfx1151 gfx1201}"

  # Experimental arches, appended to the compile lanes only, not the production
  # sweep; each also exercises the on-device HSACO load when the runner
  # matches. Empty disables.
  : "${ROCKE_CI_ARCHES_EXPERIMENTAL=gfx90a gfx1250}"

  # 'auto' sweeps every flavor rocKE publishes, so a new one is not left out,
  # and falls back to the known pair. Resolved per lane: rocKE is not
  # importable this early.
  : "${ROCKE_ENGINE_FLAVORS:=auto}"

  # Minimum families the byte-identity gate must cover for its pass to mean
  # anything.
  : "${ROCKE_MIN_GATE_FAMILIES:=20}"

  # Lanes 'all' runs, in registry order; override to scope a run, e.g.
  # 'engine ctest'.
  : "${ROCKE_ALL_LANES:=${LaneOrder[*]}}"

  # Run-shape knobs. The nightly wrappers pin both gates to 1; by hand they
  # default off, so a rerun neither wipes the lane build dir nor touches the
  # shared checkout.
  : "${ROCKE_REBUILD:=0}"
  : "${ROCKE_UPDATE_REPO:=0}"
  : "${ROCKE_SETUP_VENV:=1}"
  : "${ROCKE_REPO_URL:=https://github.com/ROCm/rocm-libraries.git}"
  : "${ROCKE_REPO_BRANCH:=develop}"
  : "${ROCKE_TORCH_INDEX_URL:=}"

  # Installing a numeric reference is host preparation, not part of a test run:
  # the index a toolchain's ROCm implies is rarely published. Opt in for a
  # one-off.
  : "${ROCKE_PROVISION_TORCH:=0}"

  # Set on the host that certifies numerics: there a missing reference is a
  # defect and reddens; elsewhere it is an unprepared host and stays an
  # unmeasured row.
  : "${ROCKE_NUMERIC_HOST:=0}"

  # rocKE uses these without declaring them in its dev extras.
  : "${ROCKE_EXTRA_TEST_DEPS:=pyarrow}"

  # Run identity and engine-extension rebuild stamp. Epoch seconds, not a bare
  # PID: PIDs repeat, and a reused one would skip a demanded rebuild; only a
  # child inherits.
  if [[ -z "${ROCKE_INTERNAL_PARENT_PID:-}" \
    || "${ROCKE_INTERNAL_PARENT_PID}" != "${PPID}" ]]; then
    ROCKE_RUN_ID=""
  fi
  export ROCKE_RUN_ID="${ROCKE_RUN_ID:-$(date +%s)-${BASHPID}}"

  # A concurrent run holds the source lock for its whole duration, so the tree
  # cannot change under a test in flight; wait a nightly's worth, then fail
  # rather than hang.
  : "${ROCKE_LOCK_WAIT:=3600}"

  # Reaches an arithmetic context, where bash would evaluate anything it is
  # given.
  [[ "${ROCKE_LOCK_WAIT}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: ROCKE_LOCK_WAIT must be a whole number of seconds" >&2
    exit 2
  }

  # The interop lanes need deterministic reference IR; the C++ engine is
  # byte-identical but not built here, so use the Python backend and suppress
  # the fallback warning.
  export ROCKE_BACKEND="${ROCKE_BACKEND:-python}"
  export PYTHONPATH="${ROCKE_TOP}/python:${ROCKE_PROJECT_ROOT}/library"
  export PYTHONPATH+="${InheritedPythonPath:+:${InheritedPythonPath}}"
}

# The state a run accumulates, set up before anything can write to it. Shared
# with the lane and environment modules, so that sourcing one has no effect.
function initRunState {
  BuildRoot="$(realpath -m "${ROCKE_CI_BUILD_ROOT}/${Stage}")"
  PyBin=""

  # Every row emitted in the run is copied here, by the bash and Python row
  # emitters alike, so the closing summary can tally without capturing its own
  # stdout.
  RowLog=""

  # Directory locks avoid inheritable file descriptors, so an external tool
  # (or a daemon it starts) cannot retain a nightly lock after this worker
  # exits.
  HeldLockDirs=()

  # shellcheck disable=SC2034
  EngineExtDir=""
  # shellcheck disable=SC2034
  EngineFlavorList=""
  Names=(); Pass=(); Tot=(); Secs=(); Skip=(); Fails=()
}

# --- Result rows

# Canonical result row for the extractor; '|' is the field separator, so strip
# it from caller-supplied fields. Relevance says how a red row should be
# triaged.
function emitRockeResult {
  local Group="${1//|/ }" Subtest="${2//|/ }" Status="${3}" Message="${4//|/ }"
  local Relevance="${5:-harness}" Row
  Row="ROCKE_RESULT|${Group}|${Subtest}|${Status}|${Message}|${Relevance//|/ }"
  echo "${Row}"
  if [[ -n "${RowLog}" ]]; then printf '%s\n' "${Row}" >> "${RowLog}"; fi
}

# A setup failure is a red data row, not a harness crash, so still exit 0.
function failSetup {
  echo "Error: ${1}"
  emitRockeResult setup "${2}" 1 "${1}"
  exit 0
}

# --- Process hygiene

# Release the locks this process still owns and drop its temporaries.
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
  [[ -n "${ToolchainShim}" ]] && rm -rf "${ToolchainShim}"
  return 0
}

# Take a directory lock, waiting for a live owner and reclaiming one left by a
# dead process, so two runs cannot share a build or a virtual environment.
function acquireDirLock {  # <lock-dir> <description>
  local Lock="${1}" Description="${2}" OwnerPid Entry Stale Waited=0 Announced=0
  while ! mkdir "${Lock}" 2>/dev/null; do
    [[ ! -L "${Lock}" ]] \
      || failSetup "refusing symlinked ${Description} lock: ${Lock}" lock
    (( Waited < ROCKE_LOCK_WAIT )) || failSetup \
      "gave up after ${ROCKE_LOCK_WAIT}s waiting for the \
${Description} lock: ${Lock}" \
      lock
    OwnerPid="$(cat "${Lock}/owner" 2>/dev/null || true)"

    # /proc, not kill -0: kill reports EPERM for another user's live process,
    # and treating that as dead would let two runs into the same tree.
    if [[ "${OwnerPid}" =~ ^[0-9]+$ && -d "/proc/${OwnerPid}" ]]; then
      # Say it once: an unexplained wait looks like a hang.
      (( Announced )) || echo "waiting for the ${Description} lock held by" \
        "pid ${OwnerPid}: ${Lock}"
      Announced=1
      sleep 1; (( ++Waited ))
      continue
    fi

    if [[ ! "${OwnerPid}" =~ ^[0-9]+$ ]]; then
      sleep 1; (( ++Waited ))
      [[ ! -e "${Lock}" ]] && continue
      OwnerPid="$(cat "${Lock}/owner" 2>/dev/null || true)"
      [[ "${OwnerPid}" =~ ^[0-9]+$ ]] || failSetup \
        "malformed ${Description} lock owner: ${Lock}" lock
      continue
    fi

    Entry="$(find "${Lock}" -mindepth 1 -maxdepth 1 ! -name owner \
      -print -quit 2>/dev/null)"
    if [[ -d "${Lock}" && -z "${Entry}" ]]; then
      Stale="${Lock}.stale.${BASHPID}"
      [[ ! -e "${Stale}" ]] \
        || failSetup "stale-lock quarantine already exists: ${Stale}" lock
      if mv "${Lock}" "${Stale}" 2>/dev/null; then
        rm -f "${Stale}/owner"
        rmdir "${Stale}" 2>/dev/null \
          || failSetup "cannot remove stale ${Description} lock: ${Stale}" lock
        continue
      fi
      # Lost the race, or the directory is not ours to move: retry as for a
      # live owner.
      sleep 1; (( ++Waited ))
      continue
    fi

    failSetup "cannot recover stale ${Description} lock: ${Lock}" lock
  done

  printf '%s\n' "${BASHPID}" > "${Lock}/owner" \
    || failSetup "cannot record ${Description} lock owner: ${Lock}" lock
  HeldLockDirs+=("${Lock}")
}

# --- The rocKE checkout

# branch@shortsha of the rocKE checkout, or '?' when it is not a git tree.
function readRockeSrcRev {
  local Branch Sha
  Branch="$(git -C "${ROCKE_TOP}" rev-parse --abbrev-ref HEAD 2>/dev/null \
    || echo '?')"
  Sha="$(git -C "${ROCKE_TOP}" rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "${Branch}@${Sha}"
}

# Make the shared rocm-libraries checkout usable before testing. rocKE and CK
# share the tree, so a refresh must not discard other work: clone only into a
# missing or empty path, require a clean checkout, and advance by fast-forward
# only.
function updateRockeSource {
  local Top Origin SourceLock
  local Repo="${ROCKE_REPO_ROOT}"
  local Url="${ROCKE_REPO_URL}"
  local Branch="${ROCKE_REPO_BRANCH}"

  if [[ "${Repo}" != "${ROCKE_TOP}" ]]; then
    mkdir -p "$(dirname "${Repo}")" \
      || failSetup "cannot create source parent: $(dirname "${Repo}")" source
    SourceLock="${Repo}.rocke-ci.lock.d"
    acquireDirLock "${SourceLock}" "rocKE source"
  fi

  if [[ "${Repo}" == "${ROCKE_TOP}" ]]; then
    echo "WARN: cannot derive the rocm-libraries root from a custom ROCKE_TOP"
  elif ! git -C "${Repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [[ -d "${Repo}" && -n "$(find "${Repo}" -mindepth 1 -maxdepth 1 \
      -print -quit 2>/dev/null)" ]]; then
      failSetup "refusing to replace non-empty non-git source path: ${Repo}" \
        source
    fi
    echo "no rocKE checkout under ${ROCKE_TOP};" \
      "cloning rocm-libraries (several GB,"
    echo "shared with CK) into ${Repo}." \
      "Point ROCKE_TOP at an existing checkout to skip."
    rmdir "${Repo}" 2>/dev/null || true
    git clone --single-branch --depth 1 -b "${Branch}" "${Url}" "${Repo}" \
      || failSetup "rocm-libraries clone failed: ${Url} (${Branch})" source
  else
    Top="$(realpath -m "$(git -C "${Repo}" rev-parse --show-toplevel)")"
    [[ "${Top}" == "$(realpath -m "${Repo}")" ]] \
      || failSetup \
        "source path is nested in another repository (${Top}); \
refusing to update: ${Repo}" \
        source

    if [[ "${ROCKE_UPDATE_REPO}" == 1 ]]; then
      git check-ref-format --branch "${Branch}" >/dev/null 2>&1 \
        || failSetup "invalid ROCKE_REPO_BRANCH: ${Branch}" source

      Origin="$(git -C "${Repo}" remote get-url origin 2>/dev/null || true)"
      [[ "${Origin}" == "${Url}" ]] \
        || failSetup \
          "source origin mismatch: expected ${Url}, found ${Origin:-<none>}" \
          source

      if [[ -n "$(git -C "${Repo}" status --porcelain)" ]]; then
        failSetup \
          "rocm-libraries checkout has local changes; \
refusing to update: ${Repo}" \
          source
      fi

      echo "updating rocm-libraries (${Repo})"
      git -C "${Repo}" fetch --prune origin \
        "+refs/heads/${Branch}:refs/remotes/origin/${Branch}" \
        || failSetup \
          "failed to fetch origin/${Branch}; refusing to test stale source" \
          source

      if git -C "${Repo}" show-ref --verify --quiet "refs/heads/${Branch}"; then
        git -C "${Repo}" merge-base --is-ancestor \
          "${Branch}" "origin/${Branch}" \
          || failSetup \
            "local ${Branch} is not a fast-forward of origin/${Branch}; \
refusing to rewrite it" \
            source
        git -C "${Repo}" switch "${Branch}" \
          || failSetup "failed to switch to source branch ${Branch}" source
      else
        git -C "${Repo}" switch --track -c "${Branch}" "origin/${Branch}" \
          || failSetup "failed to create tracking branch ${Branch}" source
      fi

      git -C "${Repo}" merge --ff-only "origin/${Branch}" \
        || failSetup \
          "failed to fast-forward ${Branch}; refusing to test stale source" \
          source
    fi
  fi
}

# --- Run context

# Refuse to run without the compiler and the worker modules.
function assertPrerequisites {
  [[ -e "${AOMP}/bin/clang++" ]] \
    || failSetup "compiler toolchain not found: ${AOMP}/bin/clang++" compiler
  [[ -f "${HelperDir}/rocke_result.py" ]] \
    || failSetup "worker modules not found: ${HelperDir}" helpers
}

# Establish the context a lane runs in, once per invocation, and report it.
function establishRunContext {
  # Context is established once per invocation: banner, source refresh,
  # hygiene gate.
  # Only a direct child of a validated 'all' process may inherit that verdict.
  InternalAllChild=0
  if [[ -n "${ROCKE_INTERNAL_PARENT_PID:-}" \
    && "${ROCKE_INTERNAL_PARENT_PID}" == "${PPID}" \
    && "$(tr '\0' ' ' < "/proc/${PPID}/cmdline" 2>/dev/null)" \
      == *run_rocke.sh*" all"* ]]; then
    InternalAllChild=1
  else
    unset ROCKE_INTERNAL_PARENT_PID
  fi

  # Runs ahead of every registry consumer, the hygiene gate included: it reads
  # each lane's required tools, so an unvalidated table would decide what gets
  # checked.
  assertLaneRegistry

  if (( InternalAllChild == 0 )); then
    printBanner
    updateRockeSource
    [[ -d "${ROCKE_TOP}/python/rocke" ]] \
      || failSetup "rocKE source not found: ${ROCKE_TOP}" source
    setupPython
    installToolchainShim
    resolveToolchainFlavor
    assertToolchainProvenance
    printProvenance

    # Last, so a log opens with one description of the run before any result
    # row.
    reportArchDrift
  else
    setupPython
    installToolchainShim
  fi
}

# Open the row log this lane tallies from.
function openRowLog {
  # A directly invoked lane tallies its own rows. The 'all' parent tallies each
  # child's log instead, and a child leaves the summary to its parent, so
  # neither needs one.
  if (( InternalAllChild == 0 )) && [[ "${Stage}" != all ]]; then
    RowLog="$(mktemp "${TMPDIR:-/tmp}/rocke-rows-${Stage}.XXXXXX")" || RowLog=""
    export ROCKE_ROW_LOG="${RowLog}"
  fi
}

# --- Running the lanes

# Every lane runs under a marked, dedicated build root: validate and mark it
# before any recursive deletion, so a typo or a symlinked stage path deletes
# nothing else.
function prepareBuildRoot {
  local Marker="${ROCKE_CI_BUILD_ROOT}/.rocke-ci-root" BuildLock
  [[ "${BuildRoot}" == "${ROCKE_CI_BUILD_ROOT}/"* ]] \
    || failSetup "build path escapes ROCKE_CI_BUILD_ROOT: ${BuildRoot}" \
      build-root

  if [[ -e "${ROCKE_CI_BUILD_ROOT}" && ! -d "${ROCKE_CI_BUILD_ROOT}" ]]; then
    failSetup \
      "ROCKE_CI_BUILD_ROOT is not a directory: ${ROCKE_CI_BUILD_ROOT}" \
      build-root
  fi

  if [[ ! -d "${ROCKE_CI_BUILD_ROOT}" ]]; then
    mkdir -p "${ROCKE_CI_BUILD_ROOT}" \
      || failSetup "cannot create build root: ${ROCKE_CI_BUILD_ROOT}" build-root
  fi

  if [[ ! -f "${Marker}" ]]; then
    if [[ -n "$(find "${ROCKE_CI_BUILD_ROOT}" \
      -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      failSetup \
        "unmarked non-empty ROCKE_CI_BUILD_ROOT; \
refusing recursive cleanup: ${ROCKE_CI_BUILD_ROOT}" \
        build-root
    fi
    printf 'rocKE CI build root\n' > "${Marker}" \
      || failSetup "cannot mark build root: ${ROCKE_CI_BUILD_ROOT}" build-root
  fi

  grep -qxF 'rocKE CI build root' "${Marker}" \
    || failSetup "invalid build-root marker: ${Marker}" build-root

  BuildLock="${ROCKE_CI_BUILD_ROOT}/.${Stage}.lock.d"
  acquireDirLock "${BuildLock}" "lane build"
  if [[ "${ROCKE_REBUILD}" == 1 ]]; then rm -rf "${BuildRoot}"; fi
  mkdir -p "${BuildRoot}" \
    || failSetup "cannot create lane build dir: ${BuildRoot}" build-root

  # Keep comgr's diagnostics: an in-process compile failure is otherwise opaque.
  export AMD_COMGR_REDIRECT_LOGS="${BuildRoot}/comgr.log"
}

# Run every lane of 'all' as a child, then print the consolidated summary.
function runAllLanes {
  # The 'all' meta-stage runs every lane in one session for one consolidated
  # report. Each lane is a child with its own build dir; a lane failure does
  # not stop the rest.
    export ROCKE_INTERNAL_PARENT_PID="${BASHPID}"
    Rc=0

    # shellcheck disable=SC2086 # intended word splitting of the lane list
    for Lane in ${ROCKE_ALL_LANES}; do
      # A nested 'all' in the lane list would re-enter here and fork unbounded.
      [[ "${Lane}" == all ]] && {
        echo "WARN: skipping nested 'all' in ROCKE_ALL_LANES"
        continue
      }

      echo "########## lane: ${Lane} ##########"
      LaneLog="$(mktemp)"; LaneStart="${SECONDS}"

      # Absolute, cd-resolved path, so a PATH-launched parent still finds the
      # child; tee keeps the live stream while the lane's rows are tallied.
      "${ScriptDir}/run_rocke.sh" "${Lane}" 2>&1 | tee "${LaneLog}"
      LaneRc="${PIPESTATUS[0]}"
      if (( LaneRc != 0 )); then
        Rc=1
        emitRockeResult setup "lane-${Lane}" 1 \
          "lane exited with status ${LaneRc}" \
          | tee -a "${LaneLog}"
      elif ! grep -aq '^ROCKE_RESULT|' "${LaneLog}"; then
        Rc=1
        emitRockeResult setup "lane-${Lane}" 1 "lane produced no result rows" \
          | tee -a "${LaneLog}"
      fi

      assertRowFloor "${Lane}" "${LaneLog}" "$(readLaneRelevance "${Lane}")" \
        | tee -a "${LaneLog}"
      absorbLaneLog "${Lane}" "${LaneLog}" "$(( SECONDS - LaneStart ))"
      rm -f "${LaneLog}"
    done

    if (( ${#Names[@]} == 0 )); then
      Msg="ROCKE_ALL_LANES selected no lanes"
      emitRockeResult setup lanes 1 "${Msg}"
      Names=(all); Pass=(0); Tot=(1); Secs=(0)
      Fails=("all|setup::lanes|harness|${Msg}")
      Rc=1
    fi

    printRunSummary
    exit "${Rc}"
}

# Close the lane: check its row floor, then tally what it produced.
function finishLane {

  # Triage class for this lane's rows that carry no per-test evidence of their
  # own.

  # Derived from the lane name, not stored, so a lane can only run its own body.

  # Before the tally reads the same log, so a floor breach is counted like any
  # red row.
  [[ -n "${RowLog}" ]] \
    && assertRowFloor "${Stage}" "${RowLog}" "${LaneRelevance}"

  echo "rocKE ${Stage} done: $(date '+%Y-%m-%d %H:%M:%S')"
  if [[ -n "${RowLog}" ]]; then
    absorbLaneLog "${Stage}" "${RowLog}" "${SECONDS}"
    printRunSummary
  fi
}

# --- Reporting

# Fold one lane's rows into the run summary. Unmeasured rows are counted too,
# so a lane that certified nothing does not read like one that certified
# everything.
function absorbLaneLog {  # <lane> <row-log> <elapsed-seconds>
  local Lane="${1}" Log="${2}" Elapsed="${3}" P T S Group Subtest Msg Tier

  # awk cannot import the unmeasured-row status, so it is spelled out here.
  # Match it before the arithmetic: "Check"+0 is 0, which would count as a
  # pass.
  read -r P T S < <(awk -F'|' \
    '/^ROCKE_RESULT\|/ {t++; if ($4=="Check") s++; else if ($4+0==0) p++}
     END {print p+0, t+0, s+0}' "${Log}")
  Names+=("${Lane}"); Pass+=("${P}"); Tot+=("${T}")
  Secs+=("${Elapsed}"); Skip+=("${S}")
  while IFS='|' read -r _ Group Subtest _ Msg Tier; do
    Fails+=("${Lane}|${Group}::${Subtest}|${Tier}|${Msg%%$'\x1f'*}")
  done < <(awk -F'|' '/^ROCKE_RESULT\|/ && $4+0!=0' "${Log}")
}

# Closing tally for whoever is watching the run. '#=' cannot collide with the
# ROCKE_RESULT contract, so the extractor and its dashboard breakdown are
# unaffected.
function printRunSummary {
  local i SumP=0 SumT=0 Note Fail FLane FRest FLoc FTier FReason
  echo "#= rocKE ${Stage} summary  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "#= origin: [rocKE] project test/tool  |  [ci-harness] a probe" \
    "this CI adds; see README"

  for (( i = 0; i < ${#Names[@]}; i++ )); do
    Note=""
    if (( Skip[i] > 0 )); then
      # The denominator is what was measurable, so unmeasured rows do not read
      # as failures; an entirely unmeasured lane says so rather than printing
      # 0 / 0.
      Note="  (${Skip[i]} not measured)"
      (( Skip[i] == Tot[i] )) \
        && Note="  (${Skip[i]} not measured -- nothing certified)"
    fi
    printf '#=   %-12s %-12s %4d / %4d  %5ds%s\n' \
      "[$(readLaneOrigin "${Names[i]}")]" "${Names[i]}" "${Pass[i]}" \
      "$(( Tot[i] - Skip[i] ))" "${Secs[i]}" "${Note}"
    SumP=$(( SumP + Pass[i] )); SumT=$(( SumT + Tot[i] - Skip[i] ))
  done

  if (( ${#Names[@]} > 1 )); then
    printf '#=   %-12s %-12s %4d / %4d\n' "" TOTAL "${SumP}" "${SumT}"
  fi

  if (( ${#Fails[@]} > 0 )); then
    echo "#= failures (relevance = can this be a toolchain regression;" \
      "see README):"
    for Fail in "${Fails[@]}"; do
      FLane="${Fail%%|*}"; FRest="${Fail#*|}"
      FLoc="${FRest%%|*}"; FRest="${FRest#*|}"
      FTier="${FRest%%|*}"; FReason="${FRest#*|}"
      # Enough to recognise the failure; the full text is in the row itself.
      if (( ${#FReason} > 80 )); then FReason="${FReason:0:77}..."; fi
      printf '#=   %-12s %-16s %-10s %s  (%s)\n' \
        "[$(readLaneOrigin "${FLane}" "${FLoc%%::*}")]" \
        "${FTier}" "${FLane}" "${FLoc}" "${FReason}"
    done
  fi
}
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

parseArguments "$@"
configureToolchain
configurePaths
configureKnobs
initRunState
trap cleanupOnExit EXIT
assertPrerequisites

# Ahead of every registry consumer, the toolchain gate included: it reads each
# lane's required tools, so an unvalidated table would decide what gets checked.
assertLaneRegistry
establishRunContext
openRowLog

# 'all' has no lane body of its own: it runs the lanes and exits.
if [[ "${Stage}" == all ]]; then
  runAllLanes
fi

prepareBuildRoot

# Triage class for this lane's rows that carry no per-test evidence of their
# own.
LaneRelevance="$(readLaneRelevance "${Stage}")"

# Derived from the lane name, not stored, so a lane can only run its own body.
"$(deriveLaneHandler "${Stage}")"

finishLane
exit 0
