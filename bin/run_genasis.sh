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
ROCM=${ROCM:-/opt/rocm}

# Use function to set and test AOMP_GPU
setaompgpu

REPO_DIR=$AOMP_REPOS_TEST/GenASis
if [ -d "$REPO_DIR" ] ; then
    echo $REPO_DIR " exists."
else
    echo "$REPO_DIR does not exist. Please run ./clone_test.sh and re-run this script."
    exit
fi

# Copy Makefile_ROCm to GenASis repository
cp Makefile_ROCm $REPO_DIR/Build/Machines/

cd $REPO_DIR

AOMP_SUPP=${AOMP_SUPP:-$HOME/local}
export OPENMPI_DIR="$AOMP_SUPP/openmpi"
export SILO_DIR="$AOMP_SUPP/silo"
export HDF5_DIR="$AOMP_SUPP/hdf5"
export GENASIS_MACHINE=ROCm

if [ -d "$AOMP" ] ; then
    echo $AOMP " exists."
else
    echo "Install latest AOMP and try again!"
    exit
fi

export GPU_ID=$($AOMP/bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1}.{2})
currdir=$(pwd)
if [ -d "$SILO_DIR" ] ; then
    echo "SILO 4.10.2 exists."
else
    echo "SILO 4.10.2 does not exist. Please run ./build_supp.sh silo in $thisdir"
    exit
fi
if [ -d "$HDF5_DIR" ] ; then
    echo "HDF5-1.12.0 exists."
    export INCLUDE_HDF5="-I$HDF5_DIR/include"
    export LIBRARY_HDF5="-L$HDF5_DIR/lib -lhdf5_fortran -lhdf5"
else
    echo "HDF5-1.12.0 does not exist. Please run ./build_supp.sh hdf5 in $thisdir"
    exit
fi
if [ -d "$OPENMPI_DIR" ] ; then
    echo "OpenMPI-4.1.1 exists."
    export MPI=${OPENMPI_DIR}
else
    echo "OpenMPI does not exist. Please run ./build_supp.sh openmpi in $thisdir"
    exit
fi

export LD_LIBRARY_PATH=$HOME/rocm/aomp/lib:$OPENMPI_DIR/lib:$LD_LIBRARY_PATH
export FORTRAN_COMPILE="$AOMP/bin/flang -c -I$OPENMPI_DIR/lib"
export CC_COMPILE="$AOMP/bin/clang"
export OTHER_LIBS="-lm -L$AOMP/lib -L$OPENMPI_DIR/lib -lmpi_usempif08 -lmpi_mpifh -lmpi -lflang -lflangmain -lflangrti -lpgmath -lomp -lomptarget "
export FORTRAN_LINK="$AOMP/bin/clang $OTHER_LIBS"
export DEVICE_COMPILE="$AOMP/bin/hipcc -D__HIP_PLATFORM_HCC__"
export HIP_DIR=$ROCM
cd $currdir
if [ "$1" != "runonly" ] ; then
  cd $REPO_DIR/Programs/UnitTests/Basics/Runtime/Executables
  echo "=================  STARTING 1st MAKE in $PWD ========"
  make PURPOSE=OPTIMIZE all
  rc=$?
  if [ $rc != 0 ] ; then
     echo "ERROR in Make"
     exit
  fi
  echo
  cd $REPO_DIR/Programs/Examples/Basics/FluidDynamics/Executables
  echo "=================  STARTING 2nd  MAKE in $PWD ========"
  make PURPOSE=OPTIMIZE all
  rc=$?
  if [ $rc != 0 ] ; then
     echo "ERROR in 2nd Make"
     exit
  fi
  echo
  cd $REPO_DIR/Programs/UnitTests/Basics/Devices/Executables
  echo "=================  STARTING 3rd  MAKE in $PWD ========"
  make PURPOSE=OPTIMIZE all
  rc=$?
  if [ $rc != 0 ] ; then
     echo "ERROR in 3rd Make"
     exit
  fi
  echo "=================  BUILD IS DONE ========"
fi

if [ "$1" != "buildonly" ] ; then
  cd $REPO_DIR/Programs/UnitTests/Basics/Runtime/Executables
  echo ./PROGRAM_HEADER_Singleton_Test_$GENASIS_MACHINE
  ./PROGRAM_HEADER_Singleton_Test_$GENASIS_MACHINE
  echo
  cd $REPO_DIR/Programs/Examples/Basics/FluidDynamics/Executables
  echo
  _cmd="./RiemannProblem_$GENASIS_MACHINE Verbosity=INFO_2 nCells=256,256 \ Dimensionality=2D FinishTime=0.25 nWrite=10"
  echo $_cmd
  time $_cmd
  echo "Done: $_cmd"
  echo
  echo "=================  attempting mpirun  ========"
  echo
  #echo ldd ./SineWaveAdvection_$GENASIS_MACHINE
  #ldd ./SineWaveAdvection_$GENASIS_MACHINE
  echo
  _cmd="$OPENMPI_DIR/bin/mpirun -np 1 $AOMP/bin/openmpi_set_cu_mask $REPO_DIR/Programs/Examples/Basics/FluidDynamics/Executables/SineWaveAdvection_$GENASIS_MACHINE nCells=128,128,128"
  echo $_cmd
  time $_cmd
  echo "=================  end mpirun  ========"
  echo "================= Now running Unitests related to offloading ========"
  cd $REPO_DIR/Programs/UnitTests/Basics/Devices/Executables
  ./AllocateDevice_Command_Test_$GENASIS_MACHINE
  echo "AllocateDevice_Command_Test_$GENASIS_MACHINE done!"
  ./AllocateHost_Command_Test_$GENASIS_MACHINE
  echo "AllocateHost_Command_Test_$GENASIS_MACHINE done!"
  ./AssociateHost_Command_Test_$GENASIS_MACHINE
  echo "AssociateHost_Command_Test_$GENASIS_MACHINE done!"
  ./DeallocateDevice_Command_Test_$GENASIS_MACHINE
  echo "DeallocateDevice_Command_Test_$GENASIS_MACHINE done!"
  ./DeviceAddress_Function_Test_$GENASIS_MACHINE
  echo "DeviceAddress_Function_Test_$GENASIS_MACHINE done!"
  ./DeviceInterface_Test_$GENASIS_MACHINE
  echo "DeviceInterface_Test_$GENASIS_MACHINE done!"
  ./DisassociateHost_Command_Test_$GENASIS_MACHINE
  echo "DisassociateHost_Command_Test_$GENASIS_MACHINE done!"
  ./UpdateDevice_Command_Test_$GENASIS_MACHINE
  echo "UpdateDevice_Command_Test_$GENASIS_MACHINE done!"
fi
