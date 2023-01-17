#!/bin/bash
# 
#  remove_patches.sh: Script to remove shared lvm-project patches
#                     This is a useful way to see your specific changes
#                     and create a new shared patch.
#
# --- Start standard header to set environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/trunk_common_vars
# --- end standard header ----

removepatch $TRUNK_REPOS/llvm-project

