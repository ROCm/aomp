#!/bin/bash
# 
#  run_sollve.sh: 
#

ulimit -t 120

AOMP_OPENMPVV_REPO_NAME=OpenMP_VV

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars
# --- end standard header ----

function make_sollve_reports(){
  # Lines for report_summary tail
  numlines=4
  if [ "$1" == "52" ]; then
    cpp_files=$(find $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME/tests/5.2 -type f -name '*cpp')
    if [ "$cpp_files" == "" ]; then
      numlines=3
    fi
  fi

  # Start reports
  make report_html
  make report_summary >> combined-results.txt
  make report_summary  | tail -"$numlines" >> abrev.combined-results.txt
  mv results_report results_report"$1"
}

# Skip unified_shared_memory and unified_address tests as they render gfx 906/900 unusable.
if [ "$SKIP_USM" == "1" ]; then
   custom_source="\" -type f ! \( -name *unified_shared_memory* -o -name *unified_address* \)\""
fi

single_case=$1
if [ $single_case ] ;then
  # escape periods for grep command
  grep_filename=`echo $single_case | sed -e 's/\./\\\./g'`
  count=`find $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME/tests -type f | grep "$grep_filename" | wc -l`
  if [ $count != 1 ] ; then
    echo "ERROR: Trying to run a single SOLLVE_VV test case:'$single_case'"
    echo "       A single unique file could not be found in $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME/tests"
    echo "       with the filename name $single_case"
    echo "       For example, try this command:"
    echo "         $0 test_target_teams_distribute_parallel_for.c"
    exit 1
  fi
  # This will get a single valid filename for SOURCES= arg on make command below.
  testsrc=`find	$AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME/tests -type f | grep "$grep_filename"`
  reldir=${testsrc#$AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME/tests/}
  this_omp_version=${reldir%%/*}
else
  make_target="all"
fi

export CXX_VERSION=""
export C_VERSION=""
export F_VERSION=""

if [ "$ROCMASTER" == "1" ]; then
  ./clone_test.sh
  pushd $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME
    # Lock at specific hash for consistency
    git reset --hard 0fbdbb9f7d3b708eb0b5458884cfbab25103d387
  popd
else
  pushd $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME
  git pull
  popd
fi

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}
FLANG=${FLANG:-flang}

# Use function to set and test AOMP_GPU
setaompgpu
if [ "${AOMP_GPU:0:3}" == "sm_" ] ; then
  triple="nvptx64"
else
  triple="amdgcn-amd-amdhsa"
fi

export MY_SOLLVE_FLAGS=${MY_SOLLVE_FLAGS:-"-O2 -fopenmp --offload-arch=$AOMP_GPU"}

pushd $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME

if [ "$make_target" == "all" ] || [ "$custom_source" != "" ] ; then
   [ -d results_report45 ] && rm -rf results_report45
   [ -d results_report50 ] && rm -rf results_report50
   [ -d results_report51 ] && rm -rf results_report51
   [ -d results_report52 ] && rm -rf results_report52
   [ -f combined-results.txt ] && rm -f combined-results.txt
   [ -f abrev.combined-results.txt ] && rm -f abrev.combined-results.txt
   make tidy
fi

# Run OpenMP 4.5 Tests
if [ "$make_target" == "all" ] ; then
  if [ "$SKIP_SOLLVE45" != 1 ]; then
    echo "--------------------------- START OMP 4.5 TESTING ---------------------"
    make CC=$AOMP/bin/clang CXX=$AOMP/bin/clang++ FC=$AOMP/bin/$FLANG CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$MY_SOLLVE_FLAGS"  OMP_VERSION=4.5 LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 all
    echo
  fi
else
  echo
  pwd
  echo
  echo "START: Single SOLLVE_VV test case: $single_case"
  echo "       Source file:  $testsrc"
  echo "       OMP_VERSION:  $this_omp_version"
  if [ -f $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME/bin/${single_case}.o ] ; then
     echo "       rm $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME/bin/${single_case}.o"
     rm $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME/bin/${single_case}.o
  fi
  if [ $this_omp_version == "5.0" ] ; then
     export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=50"
  elif [ $this_omp_version == "5.1" ] ; then
     export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=51"
  elif [ $this_omp_version == "5.2" ] ; then
     export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=52"
  elif [ $this_omp_version == "4.5" ] ; then
     export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=45"
  fi
  echo "       The full make command:"
  echo " make CC=$AOMP/bin/clang CXX=$AOMP/bin/clang++ FC=$AOMP/bin/$FLANG CFLAGS=\"-lm $MY_SOLLVE_FLAGS\" CXXFLAGS=\"$MY_SOLLVE_FLAGS\" FFLAGS=\"$MY_SOLLVE_FLAGS\" LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 OMP_VERSION=$this_omp_version SOURCES=$single_case all"
make CC=$AOMP/bin/clang CXX=$AOMP/bin/clang++ FC=$AOMP/bin/$FLANG CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$MY_SOLLVE_FLAGS" LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 OMP_VERSION=$this_omp_version SOURCES=$single_case all
  rc=$?
  echo
  echo "DONE:  Single SOLLVE_VV test case: $single_case"
  echo "       Source file:  $testsrc"
  echo "       make rc: $rc"
  if [ -f $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME/bin/${single_case}.o ] ; then
     echo "       Binary $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME/bin/${single_case}.o exists!"
     echo "       If compile worked, you may rerun the binary with this command:"
     echo " $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME/bin/${single_case}.o"
  else
     echo "       Expected binary $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME/bin/${single_case}.o does NOT exist!"
  fi
  echo
  popd
  exit $rc
fi

if [ "$SKIP_SOLLVE45" != 1 ]; then
  echo "--------------------------- OMP 4.5 Detailed Results ---------------------------" >> combined-results.txt
  echo "--------------------------- OMP 4.5 Results ---------------------------" > abrev.combined-results.txt
  make_sollve_reports 45
fi

if [ "$SKIP_SOLLVE50" != 1 ]; then
  enable_xnack=0
  if [ "$AOMP_GPU" == gfx90a ] && [ "$HSA_XNACK" == "" ]; then
    export HSA_XNACK=1
    enable_xnack=1
    echo "Turning on HSA_XNACK=1 for 5.0 to allow USM tests to pass."
  fi
  # Run OpenMP 5.0 Tests
  echo "--------------------------- START OMP 5.0 TESTING ---------------------"
  export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=50"
  make tidy
  make CC=$AOMP/bin/clang CXX=$AOMP/bin/clang++ FC=$AOMP/bin/$FLANG CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$MY_SOLLVE_FLAGS"  OMP_VERSION=5.0 LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 SOURCES="$custom_source" all
  echo
  echo "--------------------------- OMP 5.0 Detailed Results ---------------------------" >> combined-results.txt
  echo "--------------------------- OMP 5.0 Results ---------------------------" >> abrev.combined-results.txt
  make_sollve_reports 50
  if [ "$enable_xnack" == 1 ]; then
    unset HSA_XNACK
  fi
fi

if [ "$SKIP_SOLLVE51" != 1 ]; then
  enable_xnack=0
  if [ "$AOMP_GPU" == gfx90a ] && [ "$HSA_XNACK" == "" ]; then
    export HSA_XNACK=1
    enable_xnack=1
    echo "Turning on HSA_XNACK=1 for 5.0 to allow USM tests to pass."
  fi
  echo "--------------------------- START OMP 5.1 TESTING ---------------------"
  # Run OpenMP 5.1 Tests
  export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=51"

# FIXME: Tests listed here are skipped to prevent GPUs crashing. Remove the test once issue is fixed.
  custom_source="\" -type f ! \( -name *test_target_has_device_addr.c* \)\""

  make tidy
  make CC=$AOMP/bin/clang CXX=$AOMP/bin/clang++ FC=$AOMP/bin/$FLANG CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$MY_SOLLVE_FLAGS"  OMP_VERSION=5.1 LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 SOURCES="$custom_source" all
  echo
  echo "--------------------------- OMP 5.1 Detailed Results ---------------------------" >> combined-results.txt
  echo "--------------------------- OMP 5.1 Results ---------------------------" >> abrev.combined-results.txt
  make_sollve_reports 51
  custom_source=""
  if [ "$enable_xnack" == 1 ]; then
    unset HSA_XNACK
  fi
fi

if [ "$SKIP_SOLLVE52" != 1 ]; then
  enable_xnack=0
  if [ "$AOMP_GPU" == gfx90a ] && [ "$HSA_XNACK" == "" ]; then
    export HSA_XNACK=1
    enable_xnack=1
    echo "Turning on HSA_XNACK=1 for 5.0 to allow USM tests to pass."
  fi
  echo "--------------------------- START OMP 5.2 TESTING ---------------------"
  # Run OpenMP 5.2 Tests
  export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=52"
  make tidy
  make CC=$AOMP/bin/clang CXX=$AOMP/bin/clang++ FC=$AOMP/bin/$FLANG CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$MY_SOLLVE_FLAGS"  OMP_VERSION=5.2 LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 SOURCES="$custom_source" all
  echo
  echo "--------------------------- OMP 5.2 Detailed Results ---------------------------" >> combined-results.txt
  echo "--------------------------- OMP 5.2 Results ---------------------------" >> abrev.combined-results.txt
  make_sollve_reports 52
  if [ "$enable_xnack" == 1 ]; then
    unset HSA_XNACK
  fi
fi

echo "========================= ALL TESTING COMPLETE ! ====================="
echo 

cat combined-results.txt
pwd
echo
echo
echo
cat abrev.combined-results.txt
echo
popd > /dev/null
