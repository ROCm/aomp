#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
#  diff_driver.sh : show the diff between llvm driver code in trunk vs amd-staging
#     This checks both llvm-project/clang/lib/Driver and llvm-project/clang/include/clang/Driver

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

_tmpdir="/tmp/$USER"
AOMP_REPOS=${AOMP_REPOS:-/work/$USER/git/aomp20.0}

REPO_UPSTREAM=${REPO_UPSTREAM:-$AOMP_REPOS/llvm-project.upstream}
REPO_DOWNSTREAM=${REPO_DOWNSTREAM:-$AOMP_REPOS/llvm-project}

if [ ! -d $REPO_UPSTREAM ] ; then
  echo "ERROR: dir $REPO_UPSTREAM not found. Set REPO_UPSTREAM to location of upstream llvm-project"
  exit 1
fi
if [ ! -d $REPO_DOWNSTREAM ] ; then
  echo "ERROR: dir $REPO_DOWNSTREAM not found. Set REPO_DOWNSTREAM to location of llvm-project with amd-staging branch"
  exit 1
fi

dir_include="/clang/include/clang/Driver"
dir_code="/clang/lib/Driver"

mkdir -p $_tmpdir
diff -aur $REPO_UPSTREAM/$dir_include $REPO_DOWNSTREAM/$dir_include >$_tmpdir/driver_include.diff
diff -aur $REPO_UPSTREAM/$dir_code    $REPO_DOWNSTREAM/$dir_code    >$_tmpdir/driver_code.diff

echo see $_tmpdir/driver_include.diff
echo see $_tmpdir/driver_code.diff

echo wc $_tmpdir/driver_include.diff
wc $_tmpdir/driver_include.diff
echo wc $_tmpdir/driver_code.diff
wc $_tmpdir/driver_code.diff
