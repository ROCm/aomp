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

_therockdir=$TR_AOMP_REPOS/TheRock
_curdir=$PWD

mkdir -p $TR_AOMP_REPOS
if [ ! -d $TR_AOMP_REPOS ] ; then 
   echo "ERROR: $0 could not create directory $TR_AOMP_REPOS"
   exit 
fi

cd $TR_AOMP_REPOS
if [ -d $TR_AOMP_REPOS/aomp ] ; then 
   echo "WARNING:  Skipping clone of aomp , $TR_AOMP_REPOS/aomp already exists"
else
   echo git clone -b aomp-dev https://github.com/ROCm/aomp
   git clone -b aomp-dev https://github.com/ROCm/aomp
fi
if [ -d $_therockdir ] ; then 
   echo "WARNING:  Skipping clone of TheRock, $_therockdir already exists"
else
   echo git clone https://github.com/ROCm/TheRock.git
   git clone https://github.com/ROCm/TheRock.git
fi

cd $_therockdir  

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ./build_tools/fetch_sources.py 2>&1 | tee fetch_sources.out

ln -sf $TR_AOMP_REPOS/TheRock/compiler/amd-llvm $TR_AOMP_REPOS/llvm-project

# At this point we can run tr_set_aomp_branches.sh
# But for now we run tr_set_aomp_branches.sh manually  

cd $_curdir
