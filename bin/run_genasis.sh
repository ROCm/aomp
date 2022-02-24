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

# Copy Makefile_ROCm to GenASis repository
cp Makefile_ROCm $REPO_DIR/Build/Machines/

export OPENMPI_DIR="/opt/openmpi-4.0.4-newflang"
export SILO_DIR="/usr/local/silo/silo-4.10.2"
export HDF5_DIR="/usr/include/hdf5/serial"
export GENASIS_MACHINE=ROCm

if [ -d "$AOMP" ] ; then
    echo $AOMP " exists."
else
    echo $AOMP" does not exist. Installing latest AOMP..."
    mkdir -p $HOME/rocm
    cd $HOME/rocm
    if [ ! -f aomp_Ubuntu2004_14.0-2_amd64.deb ] ; then
      wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_14.0-2/aomp_Ubuntu2004_14.0-2_amd64.deb
    fi
    ar x aomp_Ubuntu2004_14.0-2_amd64.deb
    tar xvf data.tar.xz
    export AOMP=$HOME/rocm/usr/lib/aomp_14.0-2
fi
cd /tmp
if [ -d "$SILO_DIR" ] ; then
    echo "SILO 4.10.2 exists."
    export SILO_DIR=/usr/local/silo/silo-4.10.2
else
    echo "SILO 4.10.2 does not exist. Installing SILO 4.10.2..."
    cd /tmp/
    if [ ! -f silo-4.10.2.tgz ] ; then
      wget https://wci.llnl.gov/sites/wci/files/2021-01/silo-4.10.2.tgz
    fi
    tar xvf silo-4.10.2.tgz
    cd /tmp/silo-4.10.2
    mkdir -p /tmp/silo-4.10.2-install
    ./configure --prefix=$HOME/rocm/silo-4.10.2-install
    make -j8
    make install
    export SILO_DIR=$HOME/rocm/silo-4.10.2-install
fi
cd /tmp
if [ -d "$HDF5_DIR" ] ; then
    echo "HDF5-1.12.0 exists."
    export INCLUDE_HDF5="-I/usr/include/hdf5/serial"
    export LIBRARY_HDF5="-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_fortran -lhdf5"
else
    echo "HDF5-1.12.0 does not exist. Installing HDF5-1.12.0..."
    cd /tmp/
    if [ ! -f hdf5-1.12.0.tar.bz2 ] ; then
      wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.bz2
    fi
    tar xvf hdf5-1.12.0.tar.bz2
    cd /tmp/hdf5-1.12.0
    ./configure --prefix=$HOME/rocm/hdf5-1.12.0-install --enable-fortran
    make -j8
    make install
    export HDF5_DIR=$HOME/rocm/hdf5-1.12.0-install
    export INCLUDE_HDF5="-I${HDF5_DIR}/include"
    export LIBRARY_HDF5="-L${HDF5_DIR}/lib -lhdf5_fortran -lhdf5"
fi
export LIBRARY_PATH=$AOMP/lib
export OMPI_FC=flang
export OMPI_CC=clang
export OMPI_CXX=clang++
export LD_LIBRARY_PATH=$AOMP/lib:$LD_LIBRARY_PATH
export PATH=$AOMP/bin:$PATH
if [ -d "$OPENMPI_DIR" ] ; then
    echo "OpenMPI-4.0.3 exists."
    export MPI=${OPENMPI_DIR}
else
    echo "OpenMPI does not exist. Installing OpenMPI-4.1.1..."
    cd /tmp/
    if [ ! -f openmpi-4.1.1.tar.bz2 ] ; then
      wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.bz2
    fi
    tar xvf openmpi-4.1.1.tar.bz2
    cd openmpi-4.1.1
    ./configure OMPI_CC=clang OMPI_CXX=clang++ OMPI_F90=flang CXX=clang++ CC=cla
ng FC=flang --prefix=$HOME/rocm/openmpi-4.1.1-install/
    make -j8
    make install
    export OPENMPI_DIR=$HOME/rocm/openmpi-4.1.1-install
fi

export LD_LIBRARY_PATH=$HOME/rocm/aomp/lib:$OPENMPI_DIR/lib:$LD_LIBRARY_PATH
export FORTRAN_COMPILE="$AOMP/bin/flang -c -I$OPENMPI_DIR/lib"
export CC_COMPILE="$AOMP/bin/clang -c"
export OTHER_LIBS="-lm -L$AOMP/lib -L$OPENMPI_DIR/lib -lmpi_usempif08 -lmpi_mpifh -lmpi -lflang -lflangmain -lflangrti -lpgmath -lomp -lomptarget "
export FORTRAN_LINK="$AOMP/bin/clang -v $OTHER_LIBS"
export DEVICE_COMPILE="$AOMP/bin/hipcc -c -D__HIP_PLATFORM_HCC__"
export ROCM_PATH=$AOMP
cd $REPO_DIR
cd $REPO_DIR/Programs/UnitTests/Basics/Runtime/Executables

if [ "$1" != "nobuild" ] ; then
   make PURPOSE=OPTIMIZE all
   rc=$?
   if [ $rc != 0 ] ; then
      echo "ERROR in Make"
      exit
   fi
  echo
  echo "=================  BUILD IS DONE ========"
fi

  # export OMP_NUM_THREADS=8
  echo ./PROGRAM_HEADER_Singleton_Test_$GENASIS_MACHINE
  ./PROGRAM_HEADER_Singleton_Test_$GENASIS_MACHINE
  echo
  cd $REPO_DIR/Programs/Examples/Basics/FluidDynamics/Executables
  make PURPOSE=OPTIMIZE all
  echo
  _cmd="./RiemannProblem_$GENASIS_MACHINE Verbosity=INFO_2 nCells=256,256 \ Dimensionality=2D FinishTime=0.25 nWrite=10"
  echo $_cmd
  time $_cmd
  echo "Done: $_cmd"
  echo
  echo "=================  attempting mpirun  ========"
  echo
  echo ldd ./SineWaveAdvection_$GENASIS_MACHINE
  ldd ./SineWaveAdvection_$GENASIS_MACHINE
  echo
  _cmd="$OPENMPI_DIR/bin/mpirun -np 2 $REPO_DIR/Programs/Examples/Basics/FluidDynamics/Executables/SineWaveAdvection_$GENASIS_MACHINE nCells=128,128,128"
  echo $_cmd
  time $_cmd
