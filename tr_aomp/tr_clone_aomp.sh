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

#  Do TheRock initialization, 3 steps
echo "python3 -m venv .venv && source .venv/bin/activate"
python3 -m venv .venv && source .venv/bin/activate
echo "pip install -r requirements.txt"
pip install -r requirements.txt
echo "python ./build_tools/fetch_sources.py"
python ./build_tools/fetch_sources.py 2>&1 | tee fetch_sources.out

echo cd $TR_AOMP_REPOS/TheRock.orig
cd $TR_AOMP_REPOS/TheRock.orig
if [ $AOMP_BUILD_FROZEN_ROCK == 1 ] ; then 
   _shakey=`cat $thisdir/tr_aomp_hash_$AOMP_VERSION_STRING.txt`
   echo git checkout $_shakey
   git checkout $_shakey
else
   echo "WARNING: AOMP_BUILD_FROZEN_ROCK=0 is for starting new AOMP release."
   echo git checkout main
   git checkout main
   echo git pull
   git pull
   # Save the main shakey that identifies this AOMP release.
   _shakey=`git log -1 | grep commit | cut -d" " -f2`
   echo $_shakey > $thisdir/tr_aomp_hash.txt
   echo "REMINDER: Copy $thisdir/tr_aomp_hash.txt to $thisdir/tr_aomp_hash_$AOMP_VERSION_STRING.txt"
fi

# Regardless of new or frozen TheRock, AOMP needs lastest amd-staging branch of
# both llvm-project and hipify  amd-staging branch.
cd $TR_AOMP_REPOS/TheRock.orig/compiler/amd-llvm
git checkout amd-staging
git pull
cd $TR_AOMP_REPOS/TheRock.orig/compiler/hipify
git checkout amd-staging
git pull

# Copy TheRock.orig to TheRock
cd $TR_AOMP_REPOS
mkdir -p TheRock
echo rsync -a TheRock.orig/ TheRock/
rsync -a TheRock.orig/ TheRock/

# Patch TheRock with fixes needed for amd-staging
# and stable/released non-compiler branches. 
cd TheRock
if [ $AOMP_BUILD_FROZEN_ROCK == 1 ] ; then 
   echo "patch -p1 < $TR_AOMP_REPOS/aomp/tr_aomp/patches/tr_aomp_$AOMP_VERSION_STRING.patch"
   patch -p1 < $TR_AOMP_REPOS/aomp/tr_aomp/patches/tr_aomp_$AOMP_VERSION_STRING.patch
else
   echo "WARNING: AOMP_BUILD_FROZEN_ROCK=0 is for starting new AOMP release."
   echo "         Apply old AOMP release patch, correct issues, then create new patch in:"
   echo "         $TR_AOMP_REPOS/aomp/tr_aomp/patches/tr_aomp_$AOMP_VERSION_STRING.patch"
fi

# Create convenience link for developers
ln -sf $TR_AOMP_REPOS/TheRock/compiler/amd-llvm $TR_AOMP_REPOS/llvm-project

cd $_curdir
echo "DONE $0"
