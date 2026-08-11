#!/bin/bash
#
#  run_umt.sh
#
# UMT depends on a number of LLNL projects: BLT, CAMP, Conduit and Umpire

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

CC=${AOMP}/bin/clang
CXX=${AOMP}/bin/clang++
FC=${AOMP}/bin/flang

echo "AOMP    = $AOMP"
echo "AOMPHIP = $AOMPHIP"
echo "ROCM    = $ROCM"

# Use function to set and test AOMP_GPU
setaompgpu

export BLT_SRC_DIR=${BLT_SRC_DIR:-BLT}
export UMT_SRC_DIR=${UMT_SRC_DIR:-UMT}
export CAMP_SRC_DIR=${CAMP_SRC_DIR:-CAMP}
export CONDUIT_SRC_DIR=${CONDUIT_SRC_DIR:-CONDUIT}
export UMPIRE_SRC_DIR=${UMPIRE_SRC_DIR:-UMPIRE}

export AOMP_SUPP=${AOMP_SUPP:-$HOME/local}

export CMAKE=$AOMP_SUPP/cmake/bin
export MPI=$AOMP_SUPP/openmpi
export LIBRARY_PATH=$AOMP/lib:$AOMPHIP/lib:$MPI/bin:$MPI/include:$LIBRARY_PATH
export LD_LIBRARY_PATH=$AOMP/lib:$AOMPHIP/lib:$MPI/bin:$MPI/include:$LD_LIBRARY_PATH
export PATH=$MPI:$AOMP/bin:$AOMPHIP/bin:$MPI/bin:$MPI/include:$PATH

function usage(){
  echo ""
  echo "------------ Usage ---------------------"
  echo "./run_umt.sh [option]"
  echo "Options: build_umt, run_umt"
  echo "---------------------------------------"
  echo ""
}

mystat=0
allstat=0
save_status() {
    mystat=$?
    allstat=$((allstat+mystat))
    echo "status: $mystat, allstat: $allstat"
    return $mystat
}

# Clone and Build UMT and dependencies
# NOTE: May wish to add fixed release/tag versions of each repository rather
# than most recent dev branch. But catching errors as they come seems helpful
# for the moment. But need to see how much fallout from the churn there is.
if [ "$1" == "build_umt" ]; then
    echo "UMT and its dependencies will be installed in $AOMP_REPOS_TEST"

    mkdir -p $AOMP_REPOS_TEST/$BLT_SRC_DIR
    mkdir -p $AOMP_REPOS_TEST/$CAMP_SRC_DIR
    mkdir -p $AOMP_REPOS_TEST/$CONDUIT_SRC_DIR
    mkdir -p $AOMP_REPOS_TEST/$UMPIRE_SRC_DIR
    mkdir -p $AOMP_REPOS_TEST/$UMT_SRC_DIR

    # no build required for BLT
    pushd $AOMP_REPOS_TEST/$BLT_SRC_DIR
    git clone https://github.com/LLNL/blt.git .
    save_status
    popd
    if [[ $mystat -eq 0 ]]; then
        echo "PATCH SUCCESS BLT"
    else
        echo "PATCH FAILED BLT, mystat: $mystat"
        exit $mystat
    fi

    pushd $AOMP_REPOS_TEST/$CAMP_SRC_DIR
    git clone --branch v2025.12.0 https://github.com/LLNL/camp.git .
    rm -rf build
    mkdir build
    pushd build
    $CMAKE/cmake -DCMAKE_INSTALL_PREFIX=$AOMP_REPOS_TEST/$CAMP_SRC_DIR/install \
          -DBLT_SOURCE_DIR=$AOMP_REPOS_TEST/$BLT_SRC_DIR \
          -DCMAKE_C_COMPILER=$CC \
          -DCMAKE_CXX_COMPILER=$CXX \
          -DCMAKE_Fortran_COMPILER=$FC \
          ../
    make clean
    make install
    save_status
    popd
    popd

    pushd $AOMP_REPOS_TEST/$CONDUIT_SRC_DIR
    git clone --branch v0.9.6 https://github.com/LLNL/conduit.git .
    rm -rf build
    mkdir build
    pushd build
    $CMAKE/cmake ../src -DCMAKE_INSTALL_PREFIX=$AOMP_REPOS_TEST/$CONDUIT_SRC_DIR/install \
    -DBLT_SOURCE_DIR=$AOMP_REPOS_TEST/$BLT_SRC_DIR -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF \
    -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_Fortran_COMPILER=$FC \
    -DENABLE_DOCS=OFF -DENABLE_FORTRAN=ON -DENABLE_MPI=ON -DENABLE_PYTHON=OFF
    make clean
    make install
    popd
    popd

    pushd $AOMP_REPOS_TEST/$UMPIRE_SRC_DIR
    git clone https://github.com/LLNL/Umpire.git .
    git submodule update --init
    git -C src/tpl/umpire/camp checkout v2025.12.0
    rm -rf build
    mkdir build
    pushd build
    $CMAKE/cmake ../ -DCMAKE_INSTALL_PREFIX=$AOMP_REPOS_TEST/$UMPIRE_SRC_DIR/install \
    -DMPI_CXX_SKIP_MPICXX=TRUE \
    -DBUILD_SHARED_LIBS=OFF \
    -DBLT_SOURCE_DIR=$AOMP_REPOS_TEST/$BLT_SRC_DIR \
    -DMPI_Fortran_COMPILER=$MPI/bin/mpif90 \
    -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_Fortran_COMPILER=$FC \
    -DBUILD_SHARED_LIBS=OFF -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF \
    -DENABLE_DOCS=OFF -DENABLE_FORTRAN=ON -DENABLE_MPI=ON -DENABLE_HIP=ON
    make clean
    make install
    save_status
    popd
    popd

    pushd $AOMP_REPOS_TEST/$UMT_SRC_DIR
    git clone https://github.com/LLNL/UMT.git .

    # This applies specific tweaks to UMT required for Flang, we can likely
    # remove this in the near future once it's incorporated into UMT and
    # one or two smaller flang bugs are squashed
    git apply $thisdir/patches/UMT-5-9-0-amdflang-mods-with-fexceptions-disabled.patch
    save_status
    if [[ $mystat -eq 0 ]]; then
        echo "PATCH SUCCESS UMT"
    else
        echo "PATCH FAILED UMT, mystat: $mystat"
        exit $mystat
    fi

    IMPLICIT_GPU_FLANG_RT=${IMPLICIT_GPU_FLANG_RT:-1}
    if [[ $IMPLICIT_GPU_FLANG_RT -eq 1 ]]; then
        FORTRAN_OFFLOAD_LIB=
    else
        FORTRAN_OFFLOAD_LIB=$AOMP/lib/libflang_rt.hostdevice.a
    fi

    rm -rf build
    mkdir build
    pushd build

    $CMAKE/cmake ../src \
    -DCMAKE_INSTALL_PREFIX=$AOMP_REPOS_TEST/$UMT_SRC_DIR/install \
    -DCONDUIT_ROOT=$AOMP_REPOS_TEST/$CONDUIT_SRC_DIR/install \
    -DUMPIRE_ROOT=$AOMP_REPOS_TEST/$UMPIRE_SRC_DIR/install \
    -DCAMP_ROOT=$AOMP_REPOS_TEST/$CAMP_SRC_DIR/install \
    -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_Fortran_COMPILER=$FC \
    -DCMAKE_FORTRAN_OFFLOAD_LIB=$FORTRAN_OFFLOAD_LIB \
    -DCMAKE_Fortran_LINKER_WRAPPER_FLAG="-Wl," \
    -DENABLE_CUDA=OFF \
    -DENABLE_OPENMP=ON -DOPENMP_HAS_FORTRAN_INTERFACE=ON \
    -DENABLE_OPENMP_OFFLOAD=ON -DOPENMP_HAS_USE_DEVICE_ADDR=ON \
    -DHIP_ROOT_DIR=$AOMPHIP \
    -DCMAKE_HIP_PLATFORM=amd \
    -DCMAKE_HIP_ARCHITECTURES=$AOMP_GPU \
    -DENABLE_UMPIRE=TRUE

    make clean
    make install
    save_status
    popd
    popd

    if [[ $allstat -eq 0 ]]; then
        echo "BUILD SUCCESS"
    else
        echo "BUILD FAILED, allstat: $allstat"
    fi
    [[ $allstat -eq 0 ]]
    exit $?
fi

# Run UMT
if [ "$1" == "run_umt" ]; then
    $AOMP_REPOS_TEST/$UMT_SRC_DIR/install/bin/test_driver -B global -g -c 20 -u 0 -d 3,3,3 -b 2
    save_status
    $AOMP_REPOS_TEST/$UMT_SRC_DIR/install/bin/test_driver -B global -g -c 20 -u 1 -d 3,3,3 -b 2
    save_status
    $AOMP_REPOS_TEST/$UMT_SRC_DIR/install/bin/test_driver -B global -g -c 20 -u 2 -d 3,3,3 -b 2
    save_status
    $AOMP_REPOS_TEST/$UMT_SRC_DIR/install/bin/test_driver -B global -g -c 20 -u 0 -d 3,3,3 -b 1
    save_status
    $AOMP_REPOS_TEST/$UMT_SRC_DIR/install/bin/test_driver -B global -g -c 20 -u 1 -d 3,3,3 -b 1
    save_status
    $AOMP_REPOS_TEST/$UMT_SRC_DIR/install/bin/test_driver -B global -g -c 20 -u 2 -d 3,3,3 -b 1
    save_status

    $AOMP_REPOS_TEST/$UMT_SRC_DIR/install/bin/test_driver -B local -g -c 20 -u 0 -d 3,3,3 -b 2
    save_status
    $AOMP_REPOS_TEST/$UMT_SRC_DIR/install/bin/test_driver -B local -g -c 20 -u 1 -d 3,3,3 -b 2
    save_status
    $AOMP_REPOS_TEST/$UMT_SRC_DIR/install/bin/test_driver -B local -g -c 20 -u 2 -d 3,3,3 -b 2
    save_status
    $AOMP_REPOS_TEST/$UMT_SRC_DIR/install/bin/test_driver -B local -g -c 20 -u 0 -d 3,3,3 -b 1
    save_status
    $AOMP_REPOS_TEST/$UMT_SRC_DIR/install/bin/test_driver -B local -g -c 20 -u 1 -d 3,3,3 -b 1
    save_status
    $AOMP_REPOS_TEST/$UMT_SRC_DIR/install/bin/test_driver -B local -g -c 20 -u 2 -d 3,3,3 -b 1
    save_status

    if [[ $allstat -eq 0 ]]; then
        echo "TESTS PASSED"
    else
        echo "TESTS FAILED, allstat: $allstat"
    fi
    [[ $allstat -eq 0 ]]
    exit $?
fi

usage
