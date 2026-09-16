# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The Python environment, its declared dependencies, the numeric reference,
# the COD-built engine extension, and turning a runner's report into rows.
#
# Sourced by ../../run_rocke.sh, which owns the run. Sourcing defines functions and
# constant tables; it starts nothing, touches no file and cannot fail a run.
# shellcheck shell=bash
#
# The driver owns the run's state -- RocmRoot, PyBin, BuildRoot, Stage,
# LaneRelevance and the rest -- which these functions read, and in a few cases
# set for the driver to use later (setupPython assigns PyBin, engineFlavors
# assigns EngineFlavorList, the toolchain functions export the flavor knobs).
# Checked on its own, shellcheck cannot see where that state comes from, so
# SC2154 is off for the file; check the driver too, since -x follows a source
# for definitions but reports nothing inside it.
# shellcheck disable=SC2154

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
  # An era that cannot be established is not a match. Skipping the comparison when
  # either side is unknown -- a COD with no .info/version, an unparseable
  # torch.version.hip -- certified the reference without ever proving it could
  # serve as one, which is the same lie as a green row for a test that never ran.
  if [[ -z "${CodEra}" || -z "${TorchEra}" ]]; then
    echo "ERROR: cannot establish the datalayout generation of torch ROCm ${TorchVer:-?}" \
         "against COD ROCm ${Ver:-?}, so it cannot be trusted as the numeric reference"
    return 1
  fi
  if [[ "${CodEra}" != "${TorchEra}" ]]; then
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
# The wheel index a released ROCm publishes. A COD carries an unreleased ROCm, so
# the index its own version implies usually does not exist -- which is why callers
# pass a released one of the same datalayout generation (rocmEra) instead.
function rocmWheelIndex {  # <major.minor>
  [[ -n "${1}" ]] || return 1
  echo "https://download.pytorch.org/whl/rocm${1}"
}

# Install one package from a ROCm wheel index into the suite-owned venv. Separate
# from any particular package so the next one that has to match a ROCm generation
# does not grow its own copy of this.
function installRocmPackage {  # <package> <index>
  local Pkg="${1}" Idx="${2}"
  pipInstallable "${Pkg} (may be several GB)" || return 1
  echo "provisioning ${Pkg} from ${Idx}"
  "${PyBin}" -m pip install --index-url "${Idx}" "${Pkg}" || {
    echo "ERROR: no ${Pkg} at ${Idx}"
    echo "       An unreleased COD ROCm has no published wheels. Point" \
         "ROCKE_TORCH_INDEX_URL at a released"
    echo "       index of the same era ($(rocmEra "$(codRocmVersion)" || echo '?'))," \
         "e.g. $(rocmWheelIndex 7.2), or install it in ${ROCKE_VENV}."
    return 1
  }
}

# Provide the numeric reference, provisioning only when a caller asked for it.
#
# Deriving an index from the COD's own ROCm and installing gigabytes mid-run was
# the old default, and it could only succeed by luck: the run that needed it spent
# its two seconds discovering that rocm10.2 publishes nothing. A prepared host
# installs torch once, which is what both successful numeric runs did. No
# --force-reinstall either: replacing a working wheel is not this script's business.
function ensureTorch {
  validateTorch && return 0
  local Idx="${ROCKE_TORCH_INDEX_URL}"
  if [[ -z "${Idx}" ]]; then
    [[ "${ROCKE_PROVISION_TORCH}" == 1 ]] || return 1
    Idx="$(rocmWheelIndex "$(codRocmVersion)")" || {
      echo "ERROR: COD has no .info/version; set ROCKE_TORCH_INDEX_URL"
      return 1
    }
  fi
  installRocmPackage torch "${Idx}" || return 1
  validateTorch
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
  # The flavor the COD speaks, pinned from its clang's own datalayout. A check that
  # reports drift under a different flavor measured another toolchain (see the
  # converter's _DATALAYOUT_DRIFT), which is not a verdict on this COD.
  [[ "${ROCKE_CODEGEN_FLAVOR:-auto}" != auto ]] \
    && RelevanceArgs+=(--cod-flavor "${ROCKE_CODEGEN_FLAVOR}")
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
