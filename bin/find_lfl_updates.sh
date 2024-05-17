#!/bin/bash
#
#   find_lfl_updates.sh : Compare amd-stg-open directories
#   
#   to 
#
#   $AOMP_REPOS/flang/flang-legacy/17.0-4/llvm-legacy/clang/lib/Driver
#   $AOMP_REPOS/flang/flang-legacy/17.0-4/llvm-legacy/clang/include/clang/Driver
#
#   to see if there are relavent updates necessary to correct flang-legacy.
#   Many of the updates should not be made because they would change
#   the behavior of the flang-legacy driver that only exists in ASO. 
#   

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars

#

lfl_driver_dir=$AOMP_REPOS/flang/flang-legacy/17.0-4/llvm-legacy/clang/lib/Driver
lfl_driver_include_dir=$AOMP_REPOS/flang/flang-legacy/17.0-4/llvm-legacy/clang/include/clang/Driver

project_driver_dir=$AOMP_REPOS/llvm-project/clang/lib/Driver
project_driver_include_dir=$AOMP_REPOS/llvm-project/clang/include/clang/Driver

mkdir -p /tmp/$USER

echo " START: $0"
echo "        Looking for potential updates to LFL flang-legacy driver from current amd-stage-open"
echo
echo "diff -Naur $lfl_driver_dir $project_driver_dir \>/tmp/$USER/review_lfl_driver_updates.patch"
diff -Naur $lfl_driver_dir $project_driver_dir  >/tmp/$USER/review_lfl_driver_updates.patch
echo
echo "diff -Naur $lfl_driver_include_dir  $project_driver_include_dir \>/tmp/$USER/review_lfl_driver_include_updates.patch"
diff -Naur $lfl_driver_include_dir  $project_driver_include_dir >/tmp/$USER/review_lfl_driver_include_updates.patch
echo
echo
echo " DONE: $0"
echo "       README! CAREFULLY review these two diff files:"
echo
echo "       /tmp/$USER/review_lfl_driver_updates.patch"
echo "       /tmp/$USER/review_lfl_driver_include_updates.patch"
echo
echo "       Most differences should be ignored because they apply to flang-new"
echo "       OR they are not necessary in the flang-legacy binary driver which is"
echo "       built from the Last Frozen LLVM (lfl) to support the flang-legacy driver"
echo "       As more updates to trunk are made to support flang-new, there will be "
echo "       even more differences to ignore for flang-legacy"
echo
echo
