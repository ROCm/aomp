#!/bin/bash
#
#  tr_clone_aomp.sh: Clone TheRock repository to use to build aomp.sh
#                    using TheRock repo and its submodules. unlike clone_aomp.sh
#                    this script is NOT (yet) reusable to refresh all the repos. 
#
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/tr_aomp_common_vars"
# --- end standard header ----

_rockorigdir=$TR_AOMP_REPOS/TheRock.orig
_therockdir=$TR_AOMP_REPOS/TheRock
_curdir=$PWD

# tr_aomp_common_vars ensures that TR_AOMP_REPOS is set
mkdir -p $TR_AOMP_REPOS
if [ ! -d $TR_AOMP_REPOS ] ; then 
   echo "ERROR: $0 could not create directory $TR_AOMP_REPOS"
   exit 
fi

if [ -d $TR_AOMP_REPOS/aomp ] ; then 
   echo "WARNING:  Skipping clone of aomp , $TR_AOMP_REPOS/aomp already exists"
else
   echo git clone -b aomp-dev https://github.com/ROCm/aomp
   git clone -b aomp-dev https://github.com/ROCm/aomp
fi
if [ -d $_rockorigdir ] ; then 
   echo "ERROR:  $_rockorigdir already exists"
   exit 1
fi
if [ -d $_therockdir ] ; then 
   echo "ERROR:  $_therockdir already exists"
   exit 1
fi

cd $TR_AOMP_REPOS

echo git clone https://github.com/ROCm/TheRock.git -b main --remote-submodules TheRock.orig
git clone https://github.com/ROCm/TheRock.git -b main --remote-submodules TheRock.orig
cd TheRock.orig
_shakey=`cat $thisdir/tr_aomp_hash.txt`
git checkout $_shakey

#  Do TheRock initialization, 3 steps
echo "python3 -m venv .venv && source .venv/bin/activate"
python3 -m venv .venv && source .venv/bin/activate
echo "pip install -r requirements.txt"
pip install -r requirements.txt
echo "python ./build_tools/fetch_sources.py"
python ./build_tools/fetch_sources.py 2>&1 | tee fetch_sources.out

# set the AOMP branches in TheRock.orig
echo $thisdir/tr_set_aomp_branches_orig.sh
$thisdir/tr_set_aomp_branches_orig.sh

# Copy TheRock.orig to TheRock
cd $TR_AOMP_REPOS
mkdir -p TheRock
echo rsync -a TheRock.orig/ TheRock/
rsync -a TheRock.orig/ TheRock/

# Patch TheRock with fixes needed for working on current ammd-staging
# and stable/released non-compiler branches. 
cd TheRock
echo "patch -p1 < $TR_AOMP_REPOS/aomp/tr_aomp/patches/tr_aomp.patch"
patch -p1 < $TR_AOMP_REPOS/aomp/tr_aomp/patches/tr_aomp.patch

# Create convenience link for developers
ln -sf $TR_AOMP_REPOS/TheRock/compiler/amd-llvm $TR_AOMP_REPOS/llvm-project

cd $_curdir
echo "DONE $0"
