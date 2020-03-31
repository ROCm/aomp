#!/bin/bash
# 
#  run_sollve.sh: 
#
# --- Start standard header to set build environment variables ----
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

AOMP_GPU=`$AOMP/bin/mygpu`
export MY_SOLLVE_FLAGS="-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU"
patchloc=$thisdir/patches
patchdir=$AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME
patchrepo

cd $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME

if [ -d logs ] ; then 
  rm logs/*.log
fi
if [ -d results_report ] ; then 
  rm -rf results_report
fi
make clean
make CC=$AOMP/bin/clang CXX=$AOMP/bin/clang++ FC=$AOMP/bin/flang CFLAGS="-lm $MY_SOLLVE_FLAGS" CXXFLAGS="$MY_SOLLVE_FLAGS" FFLAGS="$MY_SOLLVE_FLAGS"  LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1 all


make report_html
make report_summary
removepatch
