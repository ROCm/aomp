#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
#
#  tr_create_patch.sh : Build set of patches to TheRock including submodules
#
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/tr_aomp_common_vars"
# --- end standard header ----

mkdir -p $AOMP_PATCH_DIR
cd $TR_AOMP_REPOS/TheRock
git diff --ignore-submodules . > $AOMP_PATCH_DIR/_TheRock.patch
echo "$AOMP_PATCH_DIR/_TheRock.patch"

_tmpfile=/tmp/submod$$
git submodule > $_tmpfile
while read _line ; do
   _subdir=`echo $_line | cut -d" " -f2`
   cd $TR_AOMP_REPOS/TheRock/$_subdir
   _subdirname=`echo $_subdir | tr "/" "_"`
   git diff . > $AOMP_PATCH_DIR/$_subdirname.patch
   echo "$AOMP_PATCH_DIR/$_subdirname.patch"
done < $_tmpfile
rm $_tmpfile
echo "DONE $0"
