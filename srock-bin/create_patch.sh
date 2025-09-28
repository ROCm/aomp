#!/bin/bash
#
#  tr_create_patch.sh : Build set of patches to TheRock including submodules
#
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/srock_common_vars"
# --- end standard header ----
_patch_dir=$thisdir/patches/$SROCK_COMPILER_BRANCH
echo "      mkdir -p $_patch_dir"
mkdir -p $_patch_dir
cd $SROCK_THEROCK_DIR
_patch_file=$_patch_dir/_TheRock.patch
git diff --ignore-submodules . > $_patch_file
[ ! -s "$_patch_file" ] && rm $_patch_file

_tmpfile=/tmp/submod$$
git submodule > $_tmpfile
while read _line ; do
   _subdir=`echo $_line | cut -d" " -f2`
   cd $SROCK_THEROCK_DIR/$_subdir
   _subdirname=`echo $_subdir | tr "/" "_"`
   _patch_file=$_patch_dir/$_subdirname.patch
   git diff . > $_patch_file
   [ ! -s "$_patch_file" ] && rm $_patch_file
done < $_tmpfile
rm $_tmpfile
echo "DONE $0"
