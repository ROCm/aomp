#!/bin/bash
# 
#  run_nekbone.sh: 
#
# --- Start standard header to set build environment variables ----
function getdname(){
   local __DIRN=`dirname "$1"`
   if [ "$__DIRN" = "." ] ; then
      __DIRN=$PWD;
   else
      if [ ${__DIRN:0:1} != "/" ] ; then
         if [ ${__DIRN:0:2} == ".." ] ; then
               __DIRN=`dirname $PWD`/${__DIRN:3}
         else
            if [ ${__DIRN:0:1} = "." ] ; then
               __DIRN=$PWD/${__DIRN:2}
            else
               __DIRN=$PWD/$__DIRN
            fi
         fi
      fi
   fi
   echo $__DIRN
}
thisdir=$(getdname $0)
[ ! -L "$0" ] || thisdir=$(getdname `readlink "$0"`)
. $thisdir/aomp_common_vars
# --- end standard header ----


# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}
AOMP_SUPP=${AOMP_SUPP:-$HOME/local}
AOMP_SUPP_INSTALL=${AOMP_SUPP_INSTALL:-$AOMP_SUPP/install}
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
export FFTW_DIR=$AOMP_SUPP_INSTALL/fftw-3.3.8
if [ -d "$FFTW_DIR" ] ; then
    echo "FFTW exists."
else
    echo "$FFTW_DIR does not exist. Install FFTW using ./build_supp.sh fftw and try again."
    exit
fi
export OPENMPI_DIR=$AOMP_SUPP_INSTALL/openmpi-4.1.1
if [ -d "$OPENMPI_DIR" ] ; then
    echo "OpenMPI-4.1.1 exists."
else
    echo "OpenMPI does not exist. Install FFTW3 using ./build_supp.sh ompi and try again."
    exit
fi

export LD_LIBRARY_PATH=$HOME/rocm/aomp/lib:$OPENMPI_DIR/lib:$FFTW_DIR/lib:$LD_LIBRARY_PATH
export ROCM_DIR=$AOMP
cd $REPO_DIR
if [ "$1" != "runonly" ] ; then
  echo "=================  STARTING MAKE in $PWD ========"
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
  _cmd="$OPENMPI_DIR/bin/mpirun -np 1 ./PSDNS_fft.x"
  #_cmd="gdb ./PSDNS_fft.x"
  echo $_cmd
  time $_cmd
  echo
  echo "=================  DONE with mpirun  ========"
fi
