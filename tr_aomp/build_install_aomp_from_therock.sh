#!/bin/bash
#
# build_install_aomp_from_therock.sh: build llvm-project, update TheRock dist/rocm
#                                     and install via rsync into $AOMP_INSTALL_DIR
#
# This script is intended to run after a successful tr_clone_aomp.sh 
# and tr_build_aomp.sh, and then some update is made to the llvm-project
# compiler source. 
#
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/tr_aomp_common_vars"
# --- end standard header ----
_curdir=$PWD

if [ ! -d $TR_AOMP_REPOS/TheRock/build/dist/rocm ] ; then 
  echo "ERROR: Missing directory $TR_AOMP_REPOS/TheRock/build/dist/rocm "
  echo "       Did you get a successful full build with ./tr_build_aomp.sh"
  exit 1
fi
if [ ! -d $AOMP_INSTALL_DIR ] ; then 
  echo "ERROR: Missing directory $AOMP_INSTALL_DIR"
  echo "       Did you get a successful full build with ./tr_build_aomp.sh"
  exit 1
fi
echo cd $TR_AOMP_REPOS/llvm-project/build
cd $TR_AOMP_REPOS/llvm-project/build
echo ninja therock-dist
echo
ninja therock-dist
[ $? != 0 ] && echo "ninja therock-dist FAILED" && cd $_curdir && exit 1
echo 
# verbose all all updated or deleted files 
echo "rsync -av --delete $TR_AOMP_REPOS/TheRock/build/dist/rocm/ $AOMP_INSTALL_DIR/"
rsync -av --delete $TR_AOMP_REPOS/TheRock/build/dist/rocm/ $AOMP_INSTALL_DIR/

cd $_curdir
