#!/bin/bash
# 
#  apply_patches.sh: Script to apply shared llvm-project patches.
#
# --- Start standard header to set environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/trunk_common_vars
# --- end standard header ----

patchrepo $TRUNK_REPOS/llvm-project

