#!/bin/bash
# 
#  run_genasis.sh 
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars
# --- end standard header ----

# Setup AOMP variables
AOMP=${AOMP:-$HOME/rocm/aomp/llvm}
AOMPHIP=${AOMPHIP:-$(realpath -m $(realpath -m $AOMP)/../..)}
# for ROCm utilities (e.g. rocm_agent_enumerator)
ROCM=${ROCM:-$(realpath -m $(realpath -m $AOMP)/../..)}
FLANG=${FLANG:-flang-new}

echo "AOMP    = $AOMP"
echo "AOMPHIP = $AOMPHIP"
echo "ROCM    = $ROCM"
echo "FLANG   = $FLANG"

# Use function to set and test AOMP_GPU
setaompgpu

SRC_DIR=${SRC_DIR:-GenASis}             # old source
#SRC_DIR=${SRC_DIR:-GenASiS_Basics}      # new source - uses submodules

REPO_DIR=$AOMP_REPOS_TEST/$SRC_DIR
if [ -d "$REPO_DIR" ] ; then
    echo $REPO_DIR " exists."
else
    echo "$REPO_DIR does not exist. Please run ./clone_test.sh and re-run this script."
    exit
fi

# Copy Makefile_ROCm to GenASis repository
cp Makefile_ROCmFlangNew $REPO_DIR/Build/Machines/

patchrepo $AOMP_REPOS_TEST/GenASis

cd $REPO_DIR

AOMP_SUPP=${AOMP_SUPP:-$HOME/local}
export OPENMPI_DIR="$AOMP_SUPP/openmpi"
export SILO_DIR="$AOMP_SUPP/silo"
export HDF5_DIR="$AOMP_SUPP/hdf5"
export GENASIS_MACHINE=ROCmFlangNew

if [ -d "$AOMP" ] ; then
    echo $AOMP " exists."
else
    echo "Install latest AOMP and try again!"
    exit
fi

export GPU_ID=$($ROCM/bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1}.{2})
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

export LD_LIBRARY_PATH=$AOMP/lib:$AOMPHIP/lib:$OPENMPI_DIR/lib:$LD_LIBRARY_PATH
export FORTRAN_COMPILE="$AOMP/bin/$FLANG -c -fopenmp --offload-arch=$GPU_ID -fPIC -I$OPENMPI_DIR/lib -cpp"
export CC_COMPILE="$AOMP/bin/clang -fPIC"
export OTHER_LIBS="-lm -L$AOMP/lib -lFortranRuntime -lFortranDecimal  -lomp -lomptarget -z muldefs "
export FORTRAN_LINK="$AOMP/bin/clang $OTHER_LIBS"
export DEVICE_COMPILE="$AOMPHIP/bin/hipcc -D__HIP_PLATFORM_HCC__"
export HIP_DIR=$ROCM
export HIP_CLANG_PATH=$AOMP/bin
cd $currdir
if [ "$1" != "runonly" ] ; then
  cd $REPO_DIR/Programs/UnitTests/Basics/Runtime/Executables
  echo "=================  STARTING 1st MAKE in $PWD ========"
  make clean
  make PURPOSE=OPTIMIZE all
  rc=$?
  if [ $rc != 0 ] ; then
     echo "ERROR in Make"
     exit
  fi
  echo
  cd $REPO_DIR/Programs/Examples/Basics/FluidDynamics/Executables
  echo "=================  STARTING 2nd  MAKE in $PWD ========"
  make clean
  make PURPOSE=OPTIMIZE all
  rc=$?
  if [ $rc != 0 ] ; then
     echo "ERROR in 2nd Make"
     exit
  fi
  echo
  cd $REPO_DIR/Programs/UnitTests/Basics/Devices/Executables
  echo "=================  STARTING 3rd  MAKE in $PWD ========"
  make clean
  make PURPOSE=OPTIMIZE all
  rc=$?
  if [ $rc != 0 ] ; then
     echo "ERROR in 3rd Make"
     exit
  fi
  echo "=================  BUILD IS DONE ========"
fi

if [ "$1" != "buildonly" ] ; then
  export LIBOMPTARGET_KERNEL_TRACE=1
  export LIBOMPTARGET_INFO=1
  export OMP_TARGET_OFFLOAD=MANDATORY
  cd $REPO_DIR/Programs/UnitTests/Basics/Runtime/Executables
  echo ./PROGRAM_HEADER_Singleton_Test_$GENASIS_MACHINE
  ./PROGRAM_HEADER_Singleton_Test_$GENASIS_MACHINE
  echo
  cd $REPO_DIR/Programs/Examples/Basics/FluidDynamics/Executables
  echo
  echo "=================  2D RiemannProblem ========"
  _cmd="./RiemannProblem_$GENASIS_MACHINE Verbosity=INFO_2 nCells=256,256 \ Dimensionality=2D FinishTime=0.25 nWrite=10"
  echo $_cmd
  time $_cmd
  echo "Done: $_cmd"
  echo
  echo "=================  3D RiemannProblem ========"
  _cmd="./RiemannProblem_${GENASIS_MACHINE} Verbosity=INFO_2 nCells=256,256,256 Dimensionality=3D NoWrite=T FinishCycle=10"
  echo $_cmd
  time $_cmd
  echo "Done: $_cmd"
  echo
  echo "=================  attempting mpirun  ========"
  echo
  #echo ldd ./SineWaveAdvection_$GENASIS_MACHINE
  #ldd ./SineWaveAdvection_$GENASIS_MACHINE
  echo
  _cmd="$OPENMPI_DIR/bin/mpirun -np 1 $AOMP/bin/gpurun $REPO_DIR/Programs/Examples/Basics/FluidDynamics/Executables/SineWaveAdvection_$GENASIS_MACHINE nCells=128,128,128"
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
