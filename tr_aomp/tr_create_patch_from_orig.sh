#!/bin/bash
#
#  tr_build_aomp.sh : Build aomp using TheRock 
#
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/tr_aomp_common_vars"
# --- end standard header ----

_origdir=$TR_AOMP_REPOS/orig.TheRock

mkdir -p $TR_AOMP_REPOS/aomp/tr_aomp/patches
_patchfile=$TR_AOMP_REPOS/aomp/tr_aomp/patches/tr_aomp.patch

cd $TR_AOMP_REPOS
[ -d orig.TheRock ] && echo "rm -rf orig.TheRock" && rm -rf orig.TheRock
mkdir orig.TheRock
echo rsync -a TheRock/ orig.TheRock/
rsync -a TheRock/ orig.TheRock/

cd orig.TheRock
echo "--- git status"
git status
echo "--- git status DONE"

cd $_origdir
_tmpfile=/tmp/submod$$
git submodule > $_tmpfile
while read _line ; do
  _subdir=`echo $_line | cut -d" " -f2`
  cd $_origdir/$_subdir
  if [ "$_subdir" != "compiler/amd-llvm" ] && [ "$_subdir" != "compiler/hipify" ] ; then
     echo "DIR:$PWD "
     git checkout .
     echo git checkout tr_aomp_orig_$AOMP_VERSION_STRING
     git checkout tr_aomp_orig_$AOMP_VERSION_STRING 2>/dev/null
  fi
done < $_tmpfile

echo cd $_origdir
cd $_origdir
git checkout .
echo git checkout tr_aomp_orig_$AOMP_VERSION_STRING
git checkout tr_aomp_orig_$AOMP_VERSION_STRING

rm $_tmpfile

cd $TR_AOMP_REPOS
# Important avoid development changes to llvm-project aka amd-llvm
# Also skip .git diffs and python cache 
echo "diff -aur -x .git -x amd-llvm -x hipify -x __pycache__ -x build  -x .ccache orig.TheRock TheRock > $_patchfile"
diff -aur -x .git -x amd-llvm -x hipify -x __pycache__ -x build  -x .ccache orig.TheRock TheRock > $_patchfile
