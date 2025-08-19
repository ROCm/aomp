#!/bin/bash
#
#  tr_clone_aomp.sh: Clone TheRock repository to use to build aomp.sh
#                    using TheRock repo and its submodules. unlike clone_aomp.sh
#                    this script is NOT (yet) reusable to refresh all the repos. 
#
#  WARNING:  This script is experimental. 
#

_workdir=${1:-/work}
_aomprepodir=$_workdir/$USER/git/tr_aomp
_therockdir=$_aomprepodir/TheRock
_curdir=$PWD

if [ ! -d $_workdir ] ; then 
   echo "ERROR: $0 needs directory $_workdir"
   exit 
fi

mkdir -p $_aomprepodir
if [ ! -d $_aomprepodir ] ; then 
   echo "ERROR: $0 could not create directory $_aomprepodir"
   exit 
fi

cd $_aomprepodir
if [ -d $_aomprepodir/aomp ] ; then 
   echo "WARNING:  Skipping clone of aomp , $_aomprepodir/aomp already exists"
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

# At this point we can run tr_set_aomp_branches.sh
# But for now we run tr_set_aomp_branches.sh manually  

cd $_curdir
