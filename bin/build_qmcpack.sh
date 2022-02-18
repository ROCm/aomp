#!/bin/bash
# 
#  build_qmcpack.sh: 
#
#  Users can override the following variables:
#  AOMP, AOMP_GPU, BOOST_ROOT, FFTW_HOME, OPENMPI_INSTALL, QMCPACK_REPO
#
#  This script assumes openmpi(4.0.3 recommended) is installed and built with clang, for example:
#  export LD_LIBRARY_PATH=$AOMP/lib
#  export PATH=$AOMP/bin:$PATH
#
#  ./configure --prefix=$HOME/openmpi-4.0.3-install OMPI_CC=clang OMPI_CXX=clang++ \
#  OMPI_F90=flang CXX=clang++ CC=clang FC=flang -enable-mpi1-compatibility
#
#  make all
#  make install
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
AOMP=${AOMP:-~/usr/lib/aomp}
ROCM_VER=${ROCM_VER:-rocm-alt}

if [ -e /etc/profile.d/modules.sh ] ; then
   source /etc/profile.d/modules.sh
   # This script assumes modules setup on poplar cluster
   module load cmake
   module load gcc
   module load hdf5/1.10.1
   module load openblas
   if [ "$ROCM_VER" != "none" ]; then
     module load $ROCM_VER
   fi
   BOOST_ROOT=${BOOST_ROOT:-/cm/shared/opt/boost/1.72.0}
   FFTW_HOME=${FFTW_HOME:-/cm/shared/apps/fftw/openmpi/gcc/64/3.3.8}
   OPENMPI_INSTALL=${OPENMPI_INSTALL:-~/openmpi-4.0.3-install}
   export BOOST_ROOT FFTW_HOME
else
   # Need the following packaages from ubuntu: libxml2-dev, libfftw3-dev, libboost-dev
   BOOST_ROOT=${BOOST_ROOT:-/usr/lib/x86_64-linux-gnu}
   FFTW_HOME=${FFTW_HOME:-/usr/lib/x86_64-linux-gnu}
   OPENMPI_INSTALL=${OPENMPI_INSTALL:-~/openmpi-4.0.3-install}
   export BOOST_ROOT FFTW_HOME
fi

# Use function to set and test AOMP_GPU
setaompgpu

build_folder=build_AOMP_offload_real_MP_$AOMP_GPU
QMCPACK_REPO=${QMCPACK_REPO:-$AOMP_REPOS_TEST/$AOMP_QMCPACK_REPO_NAME}
export PATH=$OPENMPI_INSTALL/bin:$AOMP/bin:$PATH
export LD_LIBRARY_PATH=$OPENMPI_INSTALL/lib:$LD_LIBRARY_PATH

if [ "$mygpu" == "unknown" ] ; then 
   echo "ERROR: No gpu found"
   exit 1
fi

pushd $QMCPACK_REPO
if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   if [ -d $build_folder ] ; then
      echo "FRESH START"
      echo rm -rf $build_folder
      rm -rf $build_folder
   fi
fi

# Both versions should show AOMP's clang
clang --version
mpicc --version

echo Environment Variables:
echo AOMP: $AOMP
echo AOMP_GPU: $AOMP_GPU
echo OPENMPI_INSTALL: $OPENMPI_INSTALL
echo BOOST_ROOT: $BOOST_ROOT
echo FFTW_HOME: $FFTW_HOME
echo QMCPACK_REPO: $QMCPACK_REPO
echo

if [[ ! -e $OPENMPI_INSTALL/bin/mpicc ]]; then
  echo ERROR: mpicc not found. We recommend openmpi 4.0.3. Set OPENMPI_INSTALL if openmpi not found in ~/openmpi-4.0.3-install.
  exit 1
fi

echo "###################################"
echo Building AOMP_offload_real_MP_$AOMP_GPU
echo "###################################"

mkdir -p $build_folder
pushd $build_folder

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
$AOMP_CMAKE -DCMAKE_C_COMPILER=$OPENMPI_INSTALL/bin/mpicc \
-DCMAKE_CXX_COMPILER=$OPENMPI_INSTALL/bin/mpicxx \
-DOMPI_CC=$AOMP/bin/clang -DOMPI_CXX=$AOMP/bin/clang++ \
-DQMC_MPI=1 \
-DOFFLOAD_ARCH=$AOMP_GPU \
-DCMAKE_C_FLAGS="-march=native" \
-DCMAKE_CXX_FLAGS="-march=native -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU" \
-DENABLE_OFFLOAD=ON -DOFFLOAD_TARGET="amdgcn-amd-amdhsa" \
-DENABLE_TIMERS=1 \
..
fi

echo
echo make -j$AOMP_JOB_THREADS
make -j$AOMP_JOB_THREADS

# Exit if build was not successful
if [ $? != 0 ]; then
  echo
  echo ERROR: make command failed.
  echo
  exit 1
fi

echo 
echo "DONE!  Build is in $build_folder. To test:"
echo "       cd $build_folder"
echo "       ctest -R deterministic"
echo 
popd
popd
