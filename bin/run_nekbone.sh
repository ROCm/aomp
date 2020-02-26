#!/bin/bash
# 
#  run_nekbone.sh: 
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
#patchloc=$thisdir/patches
#patchdir=$AOMP_REPOS_TEST/$AOMP_IAPPS_REPO_NAME
#patchrepo

cd $AOMP_REPOS_TEST/openmpapps/Nekbone
cd test/example3
make clean
PATH=$AOMP/bin/:$PATH ./mymakenek
LIBOMPTARGET_KERNEL_TRACE=1 ./nekbone

#removepatch
