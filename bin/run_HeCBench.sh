#!/bin/bash

# run_HeCBench.sh - runs HeCBench benchmarks in the $AOMP_REPOS_TEST dir.
# User can set RUN_OPTIONS to control what variants(openmp, hip) are selected.
#
# Verbose debug:  ./run_HeCBench.sh -v
#                 VERBOSE=1 ./run_HeCBench.sh

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars
# --- end standard header ----

VERBOSE=${VERBOSE:-0}
while [ $# -gt 0 ]; do
  case "$1" in
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [-v|--verbose]"
      echo "  RUN_OPTIONS     openmp hip (default: both)"
      echo "  HECBENCH_LIST   space-separated benchmark dirs to run"
      echo "  HECBENCH_TIMEOUT per-benchmark timeout seconds (default: 180)"
      echo "  LAUNCHER        passed to Makefile run target"
      echo "  VERBOSE=1       same as -v"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1 (try -h)"
      exit 1
      ;;
  esac
done

vlog() {
  if [ "$VERBOSE" == 1 ]; then
    echo "[verbose] $*"
  fi
}

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}
AOMPHIP=${AOMPHIP:-$AOMP}

# Use function to set and test AOMP_GPU
setaompgpu

RUN_OPTIONS=${RUN_OPTIONS:-"openmp hip"}
HECBENCH_TIMEOUT=${HECBENCH_TIMEOUT:-180}
HECBENCH_LIST=${HECBENCH_LIST:-""}
LAUNCHER=${LAUNCHER:-}

hecbench_root=$AOMP_REPOS_TEST/HeCBench
hecbench_src=$hecbench_root/src

vlog "AOMP=$AOMP"
vlog "AOMP_GPU=$AOMP_GPU"
vlog "AOMP_REPOS_TEST=$AOMP_REPOS_TEST"
vlog "hecbench_root=$hecbench_root"
vlog "hecbench_src=$hecbench_src"
vlog "PWD(before cd)=$(pwd)"

if [ -d "$hecbench_src" ]; then
  cd "$hecbench_src" || exit 1
  vlog "PWD(after cd)=$(pwd)"
elif [ -d "$hecbench_root" ]; then
  vlog "WARN: $hecbench_src missing; listing $hecbench_root:"
  if [ "$VERBOSE" == 1 ]; then
    ls -la "$hecbench_root"
  fi
  echo "ERROR: HeCBench src not found: $hecbench_src"
  exit 1
else
  echo "ERROR: HeCBench not found in $AOMP_REPOS_TEST."
  vlog "Expected: $hecbench_root"
  exit 1
fi

results=$hecbench_root/results.txt
rm -f "$results"

export PATH=$AOMP/bin:$PATH
export LD_LIBRARY_PATH=$AOMP/lib:$LD_LIBRARY_PATH

vlog "PATH=$PATH"
vlog "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
vlog "which clang++=$(which clang++ 2>/dev/null || echo not-found)"
vlog "which hipcc=$(which hipcc 2>/dev/null || echo not-found)"
vlog "which make=$(which make 2>/dev/null || echo not-found)"
vlog "results=$results"

echo RUN_OPTIONS: $RUN_OPTIONS
if [ "$VERBOSE" == 1 ]; then
  echo "VERBOSE: enabled"
fi
for option in $RUN_OPTIONS; do
  if [ "$option" == "openmp" ]; then
    suffix="-omp"
    makefile="Makefile.aomp"
  elif [ "$option" == "hip" ]; then
    suffix="-hip"
    makefile="Makefile"
  else
    echo "ERROR: Option not recognized: $option."
    exit 1
  fi

  if [ -n "$HECBENCH_LIST" ]; then
    dirs="$HECBENCH_LIST"
    vlog "Using HECBENCH_LIST ($option): $dirs"
  else
    dirs=$(find . -maxdepth 1 -type d -name "*$suffix" | sort | sed 's|^\./||')
    dir_count=$(echo "$dirs" | wc -w)
    vlog "Discovered $dir_count *$suffix dirs under $(pwd)"
    if [ "$VERBOSE" == 1 ] && [ -n "$dirs" ]; then
      vlog "Dirs: $dirs"
    fi
  fi

  if [ -z "$dirs" ]; then
    echo "WARNING: No benchmark dirs found for option=$option suffix=$suffix in $(pwd)"
    vlog "find pattern: *$suffix"
    continue
  fi

  ran=0
  skipped=0
  for d in $dirs; do
    if [ ! -d "$d" ]; then
      vlog "SKIP (not a directory): $d"
      skipped=$((skipped + 1))
      continue
    fi
    if [ ! -f "$d/$makefile" ]; then
      vlog "SKIP (missing $makefile): $d"
      skipped=$((skipped + 1))
      continue
    fi
    ran=$((ran + 1))
    echo "=== [$option] $d ===" | tee -a "$results"
    (
      cd "$d" || exit 1
      vlog "PWD=$(pwd)"
      if [ "$option" == "openmp" ]; then
        export EXTRA_CFLAGS='-fopenmp-offload-mandatory -fopenmp-target-fast'
        make_clean=(make -f "$makefile" "ARCH=$AOMP_GPU" clean)
        make_run=(make -f "$makefile" "ARCH=$AOMP_GPU" "LAUNCHER=$LAUNCHER" run)
      else
        unset EXTRA_CFLAGS
        make_clean=(make -f "$makefile" clean)
        make_run=(make -f "$makefile" "LAUNCHER=$LAUNCHER" run)
      fi
      vlog "${make_clean[*]}"
      if [ "$VERBOSE" == 1 ]; then
        "${make_clean[@]}"
      else
        "${make_clean[@]}" >/dev/null 2>&1
      fi
      vlog "${make_run[*]}"
      if [ "$VERBOSE" == 1 ]; then
        set -o pipefail
        timeout $HECBENCH_TIMEOUT "${make_run[@]}" 2>&1 | tee -a "$results"
        rc=${PIPESTATUS[0]}
        set +o pipefail
      else
        if timeout $HECBENCH_TIMEOUT "${make_run[@]}" >>"$results" 2>&1; then
          rc=0
        else
          rc=$?
        fi
      fi
      if [ $rc -eq 0 ]; then
        echo "STATUS $d: PASS" | tee -a "$results"
        if [ "$VERBOSE" == 1 ]; then
          "${make_clean[@]}"
        else
          "${make_clean[@]}" >/dev/null 2>&1
        fi
      else
        echo "STATUS $d: FAIL(rc=$rc)" | tee -a "$results"
      fi
    )
  done
  vlog "option=$option: ran=$ran skipped=$skipped"
  echo >> "$results"
done

vlog "Done. Results: $results"
