#!/bin/bash
# 
#  run_sollve.sh: 
#

ulimit -t 120

# Codebenchlite related setting:
export CBL=${CBL:-"/opt/mgc/embedded/codebench"}
export PATH=$CBL/bin/:$PATH
export LD_LIBRARY_PATH=$CBL/x86_64-none-linux-gnu/lib64:$CBL/lib:/opt/rocm/lib:$LD_LIBRARY_PATH

export MY_SOLLVE_FLAGS="-g -foffload=-march=gfx90a -m64 -fopenmp -O2"
export CBLC=x86_64-none-linux-gnu-gcc 
export CBLCXX=x86_64-none-linux-gnu-g++
export CBLF90=x86_64-none-linux-gnu-gfortran

export EFFLAGS="-ffree-form -ffree-line-length-none"


# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

single_case=$1
if [ $single_case ] ;then
  # escape periods for grep command
  grep_filename=`echo $single_case | sed -e 's/\./\\\./g'`
  count=`find $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME/tests -type f | grep "$grep_filename" | wc -l`
  if [ $count != 1 ] ; then
    echo "ERROR: Trying to run a single SOLLVE_VV test case:'$single_case'"
    echo "       A single unique file could not be found in $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME/tests"
    echo "       with the filename name $single_case"
    echo "       For example, try this command:"
    echo "         $0 test_target_teams_distribute_parallel_for.c"
    exit 1
  fi
  # This will get a single valid filename for SOURCES= arg on make command below.
  testsrc=`find	$AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME/tests -type f | grep "$grep_filename"`
  reldir=${testsrc#$AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME/tests/}
  this_omp_version=${reldir%%/*}
else
  make_target="all"
fi

export CXX_VERSION=""
export C_VERSION=""
export F_VERSION=""

if [ "$ROCMASTER" == "1" ] || [ "$EPSDB" == "1" ]; then
  ./clone_test.sh
  pushd $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME
    # Lock at specific hash for consistency
    git reset --hard 0fbdbb9f7d3b708eb0b5458884cfbab25103d387
  popd
else
  pushd $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME
  git pull
  popd
fi

if [ "$ROCMASTER" == "1" ]; then
  export AOMP=/opt/rocm/aomp
elif [ "$EPSDB" == "1" ]; then
  export AOMP=/opt/rocm/llvm
fi

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}

# Use function to set and test AOMP_GPU
setaompgpu
if [ "${AOMP_GPU:0:3}" == "sm_" ] ; then
  triple="nvptx64"
else
  triple="amdgcn-amd-amdhsa"
fi


pushd $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME

if [ "$make_target" == "all" ] ; then
   [ -d results_report45 ] && rm -rf results_report45
   [ -d results_report50 ] && rm -rf results_report50
   [ -d results_report51 ] && rm -rf results_report51
   [ -d results_report52 ] && rm -rf results_report52
   [ -f combined-results.txt ] && rm -f combined-results.txt
   make tidy
fi

# Run OpenMP 4.5 Tests
if [ "$make_target" == "all" ] ; then
  echo "--------------------------- START OMP 4.5 TESTING ---------------------"
  make CC=$CBLC CXX=$CBLCXX FC=$CBLF90 CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$EFFLAGS $MY_SOLLVE_FLAGS"  OMP_VERSION=4.5 LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 all
  echo
else
  echo
  pwd
  echo
  echo "START: Single SOLLVE_VV test case: $single_case"
  echo "       Source file:  $testsrc"
  echo "       OMP_VERSION:  $this_omp_version"
  if [ -f $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME/bin/${single_case}.o ] ; then
     echo "       rm $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME/bin/${single_case}.o"
     rm $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME/bin/${single_case}.o
  fi
#  if [ $this_omp_version == "5.0" ] ; then
#     export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=50"
#  elif [ $this_omp_version == "5.1" ] ; then
#     export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=51"
#  elif [ $this_omp_version == "4.5" ] ; then
#     export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=45"
#  fi
  echo "       The full make command:"
  echo " make CC=$CBLC CXX=$CBLCXX FC=$CBLF90 CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$EFFLAGS $MY_SOLLVE_FLAGS" LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 OMP_VERSION=$this_omp_version SOURCES=$single_case all"
make CC=$CBLC CXX=$CBLCXX FC=$CBLF90 CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$EFFLAGS $MY_SOLLVE_FLAGS" LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 OMP_VERSION=$this_omp_version SOURCES=$single_case all
  rc=$?
  echo
  echo "DONE:  Single SOLLVE_VV test case: $single_case"
  echo "       Source file:  $testsrc"
  echo "       make rc: $rc"
  if [ -f $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME/bin/${single_case}.o ] ; then
     echo "       Binary $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME/bin/${single_case}.o exists!"
     echo "       If compile worked, you may rerun the binary with this command:"
     echo " $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME/bin/${single_case}.o"
  else
     echo "       Expected binary $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME/bin/${single_case}.o does NOT exists!"
  fi
  echo
  popd
  exit $rc
fi

echo "--------------------------- OMP 4.5 Detailed Results ---------------------------" >> combined-results.txt
echo "--------------------------- OMP 4.5 Results ---------------------------" > abrev.combined-results.txt
make report_html
make report_summary >> combined-results.txt
make report_summary  | tail -5 >> abrev.combined-results.txt

mv results_report results_report45

# Run OpenMP 5.0 Tests
echo "--------------------------- START OMP 5.0 TESTING ---------------------"
#export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=50"
make tidy
make CC=$CBLC CXX=$CBLCXX FC=$CBLF90 CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$EFFLAGS $MY_SOLLVE_FLAGS"  OMP_VERSION=5.0 LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 all
echo 
echo "--------------------------- OMP 5.0 Detailed Results ---------------------------" >> combined-results.txt
echo "--------------------------- OMP 5.0 Results ---------------------------" >> abrev.combined-results.txt
make report_html
make report_summary >> combined-results.txt
make report_summary  | tail -5 >> abrev.combined-results.txt
mv results_report results_report50

if [ "$ROCMASTER" != "1" ] && [ "$EPSDB" != "1" ] && [ "$SKIP_SOLLVE51" != 1 ]; then
echo "--------------------------- START OMP 5.1 TESTING ---------------------"
# Run OpenMP 5.1 Tests
#export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=51"
make tidy
make CC=$CBLC CXX=$CBLCXX FC=$CBLF90 CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$EFFLAGS $MY_SOLLVE_FLAGS"  OMP_VERSION=5.1 LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 all
echo 
echo "--------------------------- OMP 5.1 Detailed Results ---------------------------" >> combined-results.txt
echo "--------------------------- OMP 5.1 Results ---------------------------" >> abrev.combined-results.txt
make report_html
make report_summary >> combined-results.txt
make report_summary  | tail -5 >> abrev.combined-results.txt
mv results_report results_report51
fi

if [ "$ROCMASTER" != "1" ] && [ "$EPSDB" != "1" ] && [ "$SKIP_SOLLVE51" != 1 ]; then
echo "--------------------------- START OMP 5.2 TESTING ---------------------"
# Run OpenMP 5.2 Tests
#export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=51"
make tidy
make CC=$CBLC CXX=$CBLCXX FC=$CBLF90 CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$EFFLAGS $MY_SOLLVE_FLAGS"  OMP_VERSION=5.2 LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 all
echo 
echo "--------------------------- OMP 5.2 Detailed Results ---------------------------" >> combined-results.txt
echo "--------------------------- OMP 5.2 Results ---------------------------" >> abrev.combined-results.txt
make report_html
make report_summary >> combined-results.txt
make report_summary  | tail -5 >> abrev.combined-results.txt
mv results_report results_report52
fi

echo "========================= ALL TESTING COMPLETE ! ====================="
echo 

cat combined-results.txt
pwd
cat abrev.combined-results.txt
popd

if [ "$ROCMASTER" == "1" ]; then
  ./check_sollve.sh
elif [ "$EPSDB" == "1" ]; then
  EPSDB=1 ./check_sollve.sh
fi
