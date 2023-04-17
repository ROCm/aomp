#!/bin/bash
export ROCM_PATH=/opt/rocm

export AOMP=$HOME/rocm/aomp
export UMT_PATCH=$HOME/git/aomp16.0/aomp/bin/patches
FLANG=${FLANG:-flang}

export UMT_PATH=$HOME/git/aomp-test/UMT2013-20140204

export OMPI_PATH=$HOME/git/ompi
export MPI_INSTALL_DIR=$HOME/local/ompi_clang

export PATH=$OG/bin:$PATH
export LD_LIBRARY_PATH=$OGLIBDIR/lib64:$ROCM_PATH/hsa/lib:$LD_LIBRARY_PATH

function usage(){
  echo ""
  echo "------------ Usage ---------------------"
  echo "./run_rajaperf.sh [backend] [option]"
  echo "Backends: hip, openmp"
  echo "Options:  build_mpi, build_umt, run_umt"
  echo "---------------------------------------"
  echo ""
}


# Build MPI
if [ "$1" == "build_mpi" ]; then

    if [ ! -d "$OMPI_PATH" ]; then
        echo "$OMPI_PATH does not exists. Cloning OMPI"
        git clone --recursive  https://github.com/open-mpi/ompi.git $OMPI_PATH
    fi

    pushd $OMPI_PATH
    ./autogen.pl
    ./configure --prefix=$MPI_INSTALL_DIR CC=$AOMP/bin/clang CXX=$AOMP/bin/clang++ FC=$AOMP/bin/$FLANG OMPI_CC=$AOMP/bin/clang OMPI_CXX=$AOMP/bin/clang++ OMPI_FC=$AOMP/bin/$FLANG
    make && make install
    popd
    exit 1
fi

export PATH=$MPI_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$MPI_INSTALL_DIR/lib:$LD_LIBRARY_PATH

# Build UMT
if [ "$1" == "build_umt" ]; then

    if [ ! -d "$MPI_INSTALL_DIR" ]; then
        echo "*******************************************************************************"
        echo "You need to build MPI first. Run the script with the build_mpi option to do so."
        echo "*******************************************************************************"
        exit 0;
    fi

    if [ ! -d "$UMT_PATH" ]; then
        echo "*************************************************************************************"
        echo "UMT not found. Expecting the sources in $UMT_PATH"
        echo "*************************************************************************************"
        exit 0;
    fi

    if [ ! -f "$UMT_PATH/patched" ]; then
        pushd $UMT_PATH
        patch -s -p0 < $UMT_PATCH/umt2013.patch
        touch patched
        popd
        pwd
    fi

    if [ -f "$UMT_PATH/make.defs" ]; then
        mv $UMT_PATH/make.defs $UMT_PATH/make.defs.backup
    fi

    cp -r ./make.defs.umt $UMT_PATH/make.defs
    pushd $UMT_PATH

    echo "*** Building UMT ***"
    make 
    pushd $UMT_PATH/Teton
    make SuOlsonTest
    popd

if [ -f "$UMT_PATH/make.defs.backup" ]; then
    mv $UMT_PATH/make.defs.backup $UMT_PATH/make.defs
fi

popd
exit 1
fi

# Run UMT
if [ "$1" == "run_umt" ]; then

    pushd $UMT_PATH/Teton
    mpirun -np 64 ./SuOlsonTest grid_64MPI_12x12x12.cmg 16 2 16 8 4
    popd
    exit 1
fi

usage