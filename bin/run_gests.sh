#!/bin/bash
# 
#  run_nekbone.sh: 
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars
# --- end standard header ----

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}
AOMP_SUPP=${AOMP_SUPP:-$HOME/local}
HIPFORT=${HIPFORT:-$HOME/rocm/aomp/hipfort/}

# Use function to set and test AOMP_GPU
setaompgpu

REPO_DIR=$AOMP_REPOS_TEST/GESTS

if [ -f $HOME/GESTS.tar ] && [ ! -d $REPO_DIR ] ; then
    cp $HOME/GESTS.tar $AOMP_REPOS_TEST
    cd $AOMP_REPOS_TEST
    tar xvf $HOME/GESTS.tar
fi

if [ -d $REPO_DIR ] ; then
    echo $REPO_DIR " exists."
else
    echo "Please contact Raghu Maddhipatla to get a copy of GESTS.tar and place the tar file at $HOME"
    exit
fi

cd $REPO_DIR
#patchrepo $REPO_DIR

if [ -d $HIPFORT ] ; then
    echo $HIPFORT " exists."
else
    echo "Install latest hipfort and try again!"
    exit
fi
currdir=$(pwd)
export FFTW_DIR=$AOMP_SUPP/fftw
if [ -d "$FFTW_DIR" ] ; then
    echo "FFTW exists."
else
    echo "$FFTW_DIR does not exist. Install FFTW using ./build_supp.sh fftw and try again."
    exit
fi
export OPENMPI_DIR=$AOMP_SUPP/openmpi
if [ -d "$OPENMPI_DIR" ] ; then
    echo "OpenMPI-4.1.1 exists."
else
    echo "$OPENMPI_DIR does not exist. Install openmpi using ./build_supp.sh ompi and try again."
    exit
fi

export LD_LIBRARY_PATH=$HOME/rocm/aomp/lib:$OPENMPI_DIR/lib:$FFTW_DIR/lib:$LD_LIBRARY_PATH
export ROCM_DIR=$AOMP
export HIP_CLANG_PATH=$AOMP/bin
cd $REPO_DIR
if [ "$1" != "runonly" ] ; then
  echo "=================  STARTING MAKE in $PWD ========"
  make clean
  make
  rc=$?
  if [ $rc != 0 ] ; then
     echo "ERROR in Make"
     exit
  fi
  echo
  echo "=================  BUILD IS DONE ========"
fi

if [ "$1" != "buildonly" ] ; then
  cd $REPO_DIR
  echo
  echo "=================  attempting mpirun  ========"
  _cmd="$OPENMPI_DIR/bin/mpirun -np 1 $AOMP/bin/gpurun ./PSDNS_fft.x"
  echo $_cmd
  time $_cmd
  echo
  echo "=================  DONE with mpirun  ========"
fi
