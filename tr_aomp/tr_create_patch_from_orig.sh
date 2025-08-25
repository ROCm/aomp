#!/bin/bash
#
#  tr_build_aomp.sh : Build aomp using TheRock 
#
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/tr_aomp_common_vars"
# --- end standard header ----

_therockdir=$TR_AOMP_REPOS/TheRock

_patchdir=$TR_AOMP_REPOS/aomp/tr_aomp/patches
mkdir -p $_patchdir
_patchfile=$_patchdir/tr_aomp.patch

cd $TR_AOMP_REPOS

if [ ! -d TheRock.orig ] ; then 
   echo "ERROR: missing directory $TR_AOMP_REPOS/TheRock.orig"
   exit 1
fi

# Important avoid development changes to llvm-project aka amd-llvm
# Also skip .git diffs and python cache 
echo "diff -aur -x .git -x amd-llvm -x hipify -x __pycache__ -x build  -x .ccache TheRock.orig TheRock > $_patchfile" 
diff -aur -x .git -x amd-llvm -x hipify -x __pycache__ -x build  -x .ccache TheRock.orig TheRock > $_patchfile

