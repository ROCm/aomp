#!/bin/bash

export DOMP=${DOMP:-openmpi-5.0.8}
export WORK=${WORK:-/tmp/ompi$$}
export COMP=$AOMP
export INST=${INST:-/tmp/inst$$/$DOMP-flang}

rm -rf $WORK
mkdir -p $WORK
pushd $WORK
wget -q https://download.open-mpi.org/release/open-mpi/v5.0/$DOMP.tar.gz
tar xf $DOMP.tar.gz
cd $DOMP
rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=$COMP/lib
export PATH=$COMP/bin:$PATH
../configure --prefix=$INST OMPI_CC=clang OMPI_CXX=clang++ OMPI_F90=flang CXX=clang++ CC=clang FC=flang -enable-mpi1-compatibility 2>&1 | tail
make -j 32 2>&1 | tail
LD_LIBRARY_PATH=$COMP/lib PATH=$COMP/bin:$PATH make -j 32 install 2>&1 | tail
popd
rm -rf $WORK
echo $INST
