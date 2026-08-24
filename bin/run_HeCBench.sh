#!/usr/bin/env bash

#
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier:  MIT
#

# run_HeCBench.sh - runs HeCBench benchmarks in the $AOMP_REPOS_TEST dir.
#
# Environment variables (set before running; none are required unless noted):
#
# Compiler / ROCm layout:
#   AOMP              LLVM compiler tree (clang++, libomp)
#                     if unset: /opt/rocm/lib/llvm
#                     if set: use as-is
#   ROCM_PATH         HIP/ROCm install root (hipcc, libamdhip64)
#                     if unset: realpath(AOMP/../..) (e.g. /opt/rocm)
#   AOMP_GPU          GPU arch for OpenMP builds (ARCH= in Makefile.aomp);
#                     auto-detected via rocm_agent_enumerator if unset
#
# Test tree (from aomp_common_vars):
#   AOMP_REPOS_TEST   parent of cloned HeCBench; default: $HOME/git/aomp-test
#                     expects: $AOMP_REPOS_TEST/HeCBench/src/<bench>-{omp,hip}
#
# Run control:
#   PROGRAMMING_MODELS       space-separated list of build variants to run;
#                     default: "openmp hip" (both)
#                       openmp  - src/*-omp dirs, build with Makefile.aomp
#                                 (clang++, ARCH=$AOMP_GPU)
#                       hip     - src/*-hip dirs, build with Makefile (hipcc)
#                     examples:
#                       PROGRAMMING_MODELS=openmp
#                       PROGRAMMING_MODELS="openmp hip"
#                       PROGRAMMING_MODELS=hip
#   HECBENCH_LIST     space-separated benchmark dirs to run (default: all)
#   HECBENCH_TIMEOUT  per-benchmark timeout in seconds (default: 180)
#   LAUNCHER          passed to Makefile run target (e.g. "gpurun time -p")
#
# Compiler flags:
#   EXTRA_CFLAGS      extra compiler flags (Makefile.aomp / Makefile); not set
#                     by this script — export before running, e.g.:
#                       export EXTRA_CFLAGS='-fopenmp-target-fast'

# --- Start standard header to set AOMP environment variables ----
ScriptDir=$(dirname "$(realpath "$0")")

# shellcheck disable=SC1091
. "${ScriptDir}/aomp_common_vars"

# If AOMP and ROCM_PATH are already set, use them.  If not, use defaults.
# The default for AOMP is /opt/rocm/lib/llvm.  The default for ROCM_PATH is $AOMP/../..
export AOMP="${AOMP:-/opt/rocm/lib/llvm}"
export ROCM_PATH="${ROCM_PATH:-$(realpath -m "${AOMP}/../..")}"
export PATH="${AOMP}/bin:${ROCM_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${AOMP}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH}"

HECBENCH_LIST="${HECBENCH_LIST:-}"
HECBENCH_TIMEOUT="${HECBENCH_TIMEOUT:-180}"
LAUNCHER="${LAUNCHER:-}"
PROGRAMMING_MODELS="${PROGRAMMING_MODELS:-"openmp hip"}"

HecBenchRoot="${AOMP_REPOS_TEST}/HeCBench"
HecBenchSrc="${HecBenchRoot}/src"
ResultsFile="${HecBenchRoot}/results.txt"

# Warn when hipcc and clang do not come from the same compiler build, which a
# hip benchmark would otherwise mix silently.
function checkHipccClangMismatch {
  local HipccBin ClangBin HipccClangVersion ClangVersion
  HipccBin=$(command -v hipcc 2>/dev/null)
  ClangBin=$(command -v clang 2>/dev/null)
  if [ -z "${HipccBin}" ] || [ -z "${ClangBin}" ]; then
    return 0
  fi
  HipccClangVersion=$("${HipccBin}" --version 2>&1 | grep -i 'clang version')
  ClangVersion=$("${ClangBin}" --version 2>&1 | grep -i 'clang version')
  if [ -z "${HipccClangVersion}" ] || [ -z "${ClangVersion}" ]; then
    echo "WARNING: hipcc and clang compiler versions unverified." >&2
  elif [ "${HipccClangVersion}" == "${ClangVersion}" ]; then
    echo "INFO: hipcc and clang compiler versions match." >&2
  else
    echo "WARNING: hipcc and clang report different compiler versions:" >&2
    echo "  hipcc (${HipccBin}):" >&2
    printf '    %s\n' "${HipccClangVersion}" >&2
    echo "  clang (${ClangBin}): ${ClangVersion}" >&2
  fi
}

# Build and run benchmark $1 under model $2 with makefile $3, and record the
# verdict in the results log. The body is a subshell, so the build directory
# stays inside it.
function runBenchmark {
  (
    local Dir=$1 Model=$2 MakeFile=$3
    local -a MakeClean MakeRun
    local Rc
    cd "${Dir}" || exit 1

    if [ "${Model}" == "openmp" ]; then
      MakeClean=(make -f "${MakeFile}" "ARCH=${AOMP_GPU}" clean)
      MakeRun=(make -f "${MakeFile}" "ARCH=${AOMP_GPU}" "LAUNCHER=${LAUNCHER}" run)
    else
      MakeClean=(make -f "${MakeFile}" clean)
      MakeRun=(make -f "${MakeFile}" "LAUNCHER=${LAUNCHER}" run)
    fi

    "${MakeClean[@]}" >/dev/null 2>&1
    timeout "${HECBENCH_TIMEOUT}" "${MakeRun[@]}" 2>&1 | tee -a "${ResultsFile}"
    # Gather the return code of the timeout command specifically.
    Rc=${PIPESTATUS[0]}
    if [ "${Rc}" -eq 0 ]; then
      echo "STATUS ${Dir}: PASS" | tee -a "${ResultsFile}"
      "${MakeClean[@]}" >/dev/null 2>&1
    else
      echo "STATUS ${Dir}: FAIL(rc=${Rc})" | tee -a "${ResultsFile}"
    fi
  )
}

# --- Start of the run ----
setaompgpu

if [ ! -d "${HecBenchRoot}" ]; then
  echo "ERROR: HeCBench not found in ${AOMP_REPOS_TEST}."
  exit 1
elif [ ! -d "${HecBenchSrc}" ]; then
  echo "ERROR: HeCBench src not found: ${HecBenchSrc}"
  exit 1
fi

cd "${HecBenchSrc}" || exit 1

rm -f "${ResultsFile}"

checkHipccClangMismatch

echo PROGRAMMING_MODELS: "${PROGRAMMING_MODELS}"
for Model in ${PROGRAMMING_MODELS}; do
  if [ "${Model}" == "openmp" ]; then
    Suffix="-omp"
    MakeFile="Makefile.aomp"
  elif [ "${Model}" == "hip" ]; then
    Suffix="-hip"
    MakeFile="Makefile"
  else
    echo "ERROR: Option not recognized: ${Model}."
    exit 1
  fi

  if [ -n "${HECBENCH_LIST}" ]; then
    Dirs=${HECBENCH_LIST}
  else
    Dirs=$(find . -maxdepth 1 -type d -name "*${Suffix}" | sort | sed 's|^\./||')
  fi

  if [ -z "${Dirs}" ]; then
    echo "WARNING: No benchmark dirs found for model=${Model} suffix=${Suffix} in $(pwd)"
    continue
  fi

  NumTestsRun=0
  NumTestsSkipped=0
  for Dir in ${Dirs}; do
    if [ ! -d "${Dir}" ] || [ ! -f "${Dir}/${MakeFile}" ]; then
      NumTestsSkipped=$((NumTestsSkipped + 1))
      continue
    fi
    NumTestsRun=$((NumTestsRun + 1))
    echo "=== [${Model}] ${Dir} ===" | tee -a "${ResultsFile}"
    runBenchmark "${Dir}" "${Model}" "${MakeFile}"
  done

  echo "[${Model}] NumTestsRun=${NumTestsRun} NumTestsSkipped=${NumTestsSkipped}"
  echo -e "\n=== SUMMARY [${Model}] ===" | tee -a "${ResultsFile}"
  echo "NumTestsRun=${NumTestsRun}" | tee -a "${ResultsFile}"
  echo "NumTestsSkipped=${NumTestsSkipped}" | tee -a "${ResultsFile}"
done
