#!/bin/bash
# 
#  run_openmpapps.sh: 
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars
# --- end standard header ----

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}
FLANG=${FLANG:-flang}

#export OMPIDIR=$AOMP_SUPP/openmpi
#if [ -d "$OMPIDIR" ] ; then
#  echo "OpenMPI-4.1.1 exists."
#else
#  echo "$OMPIDIR does not exist. Install openmpi using ./build_supp.sh ompi and try again."
#  exit
#fi

# Use function to set and test AOMP_GPU
setaompgpu

cd $AOMP_REPOS_TEST/openmpapps
./check_openmpapps.sh
ret=$?
exit $ret
