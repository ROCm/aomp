#!/usr/bin/env bash

#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
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
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")

# shellcheck disable=SC1091
. "$thisdir/aomp_common_vars"

# If AOMP and ROCM_PATH are already set, use them.  If not, use defaults.
# The default for AOMP is /opt/rocm/llvm.  The default for ROCM_PATH is $AOMP/../..
export AOMP="${AOMP:-/opt/rocm/lib/llvm}"
export ROCM_PATH="${ROCM_PATH:-$(realpath -m "${AOMP}/../..")}"
export PATH=$AOMP/bin:$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$AOMP/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH

PROGRAMMING_MODELS=${PROGRAMMING_MODELS:-"openmp hip"}
HECBENCH_TIMEOUT=${HECBENCH_TIMEOUT:-180}
HECBENCH_LIST=${HECBENCH_LIST:-""}
LAUNCHER=${LAUNCHER:-}

hecbench_root=$AOMP_REPOS_TEST/HeCBench
hecbench_src=$hecbench_root/src

check_hipcc_clang_mismatch() {
  local hipcc_bin clang_bin hipcc_clang_line clang_ver
  hipcc_bin=$(PATH="$AOMP/bin:$ROCM_PATH/bin:$PATH" command -v hipcc 2>/dev/null)
  clang_bin=$(PATH="$AOMP/bin:$PATH" command -v clang 2>/dev/null)
  if [ -z "$hipcc_bin" ] || [ -z "$clang_bin" ]; then
    return 0
  fi
  hipcc_clang_line=$("$hipcc_bin" --version 2>&1 | grep -i 'clang version')
  clang_ver=$("$clang_bin" --version 2>&1 | grep -i 'clang version')
  if [ -n "$hipcc_clang_line" ] && [ -n "$clang_ver" ]; then
    if [ "$hipcc_clang_line" == "$clang_ver" ]; then
      echo "INFO: hipcc and clang compiler versions match." >&2
    else
      echo "WARNING: hipcc and clang report different compiler versions:" >&2
      echo "  hipcc ($hipcc_bin):" >&2
      printf '    %s\n' "$hipcc_clang_line" >&2
      echo "  clang ($clang_bin): $clang_ver" >&2
    fi
  else
    echo "WARNING: hipcc and clang compiler versions unverified." >&2
  fi
}

# Use function to set and test AOMP_GPU
setaompgpu

if [ ! -d "$hecbench_root" ]; then
  echo "ERROR: HeCBench not found in $AOMP_REPOS_TEST."
  exit 1
elif [ ! -d "$hecbench_src" ]; then
  echo "ERROR: HeCBench src not found: $hecbench_src"
  exit 1
fi

cd "$hecbench_src" || exit 1

results=$hecbench_root/results.txt
rm -f "$results"

# Check for a mismatch.
check_hipcc_clang_mismatch

echo PROGRAMMING_MODELS: "$PROGRAMMING_MODELS"
for model in $PROGRAMMING_MODELS; do
  if [ "$model" == "openmp" ]; then
    suffix="-omp"
    makefile="Makefile.aomp"
  elif [ "$model" == "hip" ]; then
    suffix="-hip"
    makefile="Makefile"
  else
    echo "ERROR: Option not recognized: $model."
    exit 1
  fi

  if [ -n "$HECBENCH_LIST" ]; then
    dirs="$HECBENCH_LIST"
  else
    dirs=$(find . -maxdepth 1 -type d -name "*$suffix" | sort | sed 's|^\./||')
  fi

  if [ -z "$dirs" ]; then
    echo "WARNING: No benchmark dirs found for model=$model suffix=$suffix in $(pwd)"
    continue
  fi

  NumTestsRun=0
  NumTestsSkipped=0
  for d in $dirs; do
    if [ ! -d "$d" ]; then
      NumTestsSkipped=$((NumTestsSkipped + 1))
      continue
    fi
    if [ ! -f "$d/$makefile" ]; then
      NumTestsSkipped=$((NumTestsSkipped + 1))
      continue
    fi
    NumTestsRun=$((NumTestsRun + 1))
    echo "=== [$model] $d ===" | tee -a "$results"
    (
      cd "$d" || exit 1
      if [ "$model" == "openmp" ]; then
        make_clean=(make -f "$makefile" "ARCH=$AOMP_GPU" clean)
        make_run=(make -f "$makefile" "ARCH=$AOMP_GPU" "LAUNCHER=$LAUNCHER" run)
      else
        make_clean=(make -f "$makefile" clean)
        make_run=(make -f "$makefile" "LAUNCHER=$LAUNCHER" run)
      fi
      "${make_clean[@]}" >/dev/null 2>&1
      if timeout "$HECBENCH_TIMEOUT" "${make_run[@]}" >>"$results" 2>&1; then
        echo "STATUS $d: PASS" | tee -a "$results"
        "${make_clean[@]}" >/dev/null 2>&1
      else
        echo "STATUS $d: FAIL(rc=$?)" | tee -a "$results"
      fi
    )
  done
  echo "[$model] NumTestsRun=$NumTestsRun NumTestsSkipped=$NumTestsSkipped"
  echo >> "$results"
  echo "=== SUMMARY [$model] ===" | tee -a "$results"
  echo "NumTestsRun=$NumTestsRun" | tee -a "$results"
  echo "NumTestsSkipped=$NumTestsSkipped" | tee -a "$results"
done
