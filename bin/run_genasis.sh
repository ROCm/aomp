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

# Use function to set and test AOMP_GPU
setaompgpu

REPO_DIR=$AOMP_REPOS_TEST/GenASis
cd $REPO_DIR
patchrepo $REPO_DIR
cd $REPO_DIR/Programs/UnitTests/Basics/Runtime/Executables
export LD_LIBRARY_PATH=$HOME/rocm/aomp/lib:/opt/openmpi-4.0.3/lib:$LD_LIBRARY_PATH
export GENASIS_MACHINE=ROCm
make PURPOSE=OPTIMIZE all
rc=$?
if [ $rc == 0 ] ; then
  export OMP_NUM_THREADS=8
  ./PROGRAM_HEADER_Singleton_Test_ROCm
  cd $REPO_DIR/Programs/Examples/Basics/FluidDynamics/Executables
  make PURPOSE=OPTIMIZE all
  ./RiemannProblem_ROCm Verbosity=INFO_2 nCells=256,256 \ Dimensionality=2D FinishTime=0.25 nWrite=10
  #mpirun -np 8 ./SineWaveAdvection_Linux_GCC nCells=128,128,128
fi
