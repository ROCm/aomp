#! /usr/bin/env bash

# HPC2021_SOURCE_DIR           where to clone sources to. Default: AOMP_REPOS_TEST
# HPC2021_BUILD_NUM_THREADS    Number of parallel compile processes. Default: 32
# export HPC_INPUT=ref   if you wantto run reference isntead of test

export HPG_INPUT=${HPG_INPUT:-test}
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars

: ${HPC2021_SOURCE_DIR:=$AOMP_REPOS_TEST/hpc2021-1.1.9}
: ${HPC2021_BUILD_NUM_THREADS:=32}

export WORK=/tmp/ompi$$
export COMP=$AOMP
export INST=$WORK/openmpi-5.0.8-flang

rm -rf $WORK
mkdir -p $WORK
cd $WORK
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.8.tar.gz
tar xf openmpi-5.0.8.tar.gz
cd  openmpi-5.0.8
rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=$COMP/lib
export PATH=$COMP/bin:$PATH
../configure --prefix=$INST OMPI_CC=clang OMPI_CXX=clang++ OMPI_F90=flang CXX=clang++ CC=clang FC=flang -enable-mpi1-compatibility
make -j 32
LD_LIBRARY_PATH=$COMP/lib PATH=$COMP/bin:$PATH make -j 32 install


if [ "$1" == "-clean" ]; then
  rm -rf ${HPC2021_SOURCE_DIR}
  mkdir -p ${HPC2021_SOURCE_DIR}
  cd ${HPC2021_SOURCE_DIR} || exit 1
  set -x
  #WLOC=http://roclogin.amd.com/SPEC
  WLOC=https://compute-artifactory.amd.com/artifactory/rocm-generic-local/compiler-infra
  wget --timeout 15 --tries=3 -q $WLOC/hpc2021-1.1.9.tar.xz
  wget --timeout 15 --tries=3 -q $WLOC/Hpc21-scripts.tar
  tar xf hpc2021-1.1.9.tar.xz
  tar xvf Hpc21-scripts.tar
  set +x
  ./install.sh -f
else
  cd ${HPC2021_SOURCE_DIR} || exit 1
fi
export PATH=$AOMP/../bin:$AOMP/../../bin:$PATH
export MPI=$INST
./runOne
rm -rf $WORK
#grep ratio= result/*.log
