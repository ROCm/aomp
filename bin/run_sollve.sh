#!/bin/bash
# 
#  run_sollve.sh: 
#
# --- Start standard header to set build environment variables ----

ulimit -t 120

function getdname(){
   local __DIRN=`dirname "$1"`
   if [ "$__DIRN" = "." ] ; then
      __DIRN=$PWD;
   else
      if [ ${__DIRN:0:1} != "/" ] ; then
         if [ ${__DIRN:0:2} == ".." ] ; then
               __DIRN=`dirname $PWD`/${__DIRN:3}
         else
            if [ ${__DIRN:0:1} = "." ] ; then
               __DIRN=$PWD/${__DIRN:2}
            else
               __DIRN=$PWD/$__DIRN
            fi
         fi
      fi
   fi
   echo $__DIRN
}
thisdir=$(getdname $0)
[ ! -L "$0" ] || thisdir=$(getdname `readlink "$0"`)
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

if [ "$ROCMASTER" == "1" ] || [ "$EPSDB" == "1" ]; then
  ./clone_aomp_test.sh
  pushd $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME
    # Lock at specific hash for consistency
    git reset --hard 0fbdbb9f7d3b708eb0b5458884cfbab25103d387
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

export MY_SOLLVE_FLAGS="-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU"

pushd $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME

if [ "$make_target" == "all" ] ; then
rm -rf results_report45
rm -rf results_report50
rm -rf combined-results.txt

make tidy
fi

# Run OpenMP 4.5 Tests
if [ "$make_target" == "all" ] ; then
make CC=$AOMP/bin/clang CXX=$AOMP/bin/clang++ FC=$AOMP/bin/flang CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$MY_SOLLVE_FLAGS" LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 all
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
  if [ $this_omp_version == "5.0" ] ; then
     export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=50"
  elif [ $this_omp_version == "5.1" ] ; then
     export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=51"
  elif [ $this_omp_version == "4.5" ] ; then
     export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=45"
  fi
  echo "       The full make command:"
  echo " make CC=$AOMP/bin/clang CXX=$AOMP/bin/clang++ FC=$AOMP/bin/flang CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$MY_SOLLVE_FLAGS" LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 OMP_VERSION=$this_omp_version SOURCES=$single_case all"
make CC=$AOMP/bin/clang CXX=$AOMP/bin/clang++ FC=$AOMP/bin/flang CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$MY_SOLLVE_FLAGS" LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 OMP_VERSION=$this_omp_version SOURCES=$single_case all
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

echo "--------------------------- OMP 4.5 Results ---------------------------" >> combined-results.txt
echo "--------------------------- OMP 4.5 Results ---------------------------" > abrev.combined-results.txt
make report_html
make report_summary >> combined-results.txt
make report_summary  | tail -5 >> abrev.combined-results.txt

mv results_report results_report45

# Run OpenMP 5.0 Tests
export MY_SOLLVE_FLAGS="$MY_SOLLVE_FLAGS -fopenmp-version=50"
make tidy
make CC=$AOMP/bin/clang CXX=$AOMP/bin/clang++ FC=$AOMP/bin/flang CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$MY_SOLLVE_FLAGS"  OMP_VERSION=5.0 LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 all

echo "--------------------------- OMP 5.0 Results ---------------------------" >> combined-results.txt
echo "--------------------------- OMP 5.0 Results ---------------------------" >> abrev.combined-results.txt
make report_html
make report_summary >> combined-results.txt
make report_summary  | tail -5 >> abrev.combined-results.txt
mv results_report results_report50
cat combined-results.txt
pwd
cat abrev.combined-results.txt
popd

if [ "$ROCMASTER" == "1" ]; then
  ./check_sollve.sh
elif [ "$EPSDB" == "1" ]; then
  EPSDB=1 ./check_sollve.sh
fi
