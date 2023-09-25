#!/bin/bash
export AOMP=$HOME/rocm/aomp
export UMT_PATCH=$HOME/git/aomp18.0/aomp/bin/patches

export MPI_PATH=${HOME}/local/openmpi

export UMT_PATH=$HOME/git/aomp-test/UMT-GIT
export UMT_INSTALL_PATH=$UMT_PATH/umt_workspace/install

export LD_LIBRARY_PATH=${AOMP}/lib:${MPI_PATH}/lib:${LD_LIBRARY_PATH}

CC=${AOMP}/bin/clang
CXX=${AOMP}/bin/clang++
FC=${AOMP}/bin/flang

export PATH=$AOMP/bin:$PATH


function usage(){
  echo ""
  echo "------------ Usage ---------------------"
  echo "./run_umt_git.sh [backend] [option]"
  echo "Backends: hip, openmp"
  echo "Options:  clone_umt, build_umt, run_umt"
  echo "---------------------------------------"
  echo ""
}

# clone_umt
if [ "$1" == "clone_umt" ]; then

    if [ ! -d "$UMT_PATH" ]; then
        echo "$UMT_PATH does not exists. Cloning UMT."
        git clone --recursive  https://github.com/LLNL/UMT.git $UMT_PATH
        pushd ${UMT_PATH}
        git reset --hard a6e8898d34b9ea3ad6cd5603ddfd44e05c53b7ec
        popd
    fi
    exit 1;
fi


export PATH=$MPI_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$MPI_INSTALL_DIR/lib:$LD_LIBRARY_PATH

# Build UMT
if [ "$1" == "build_umt" ]; then

    if [ ! -d "$UMT_PATH" ]; then
        echo "*************************************************************************************"
        echo "UMT not found. Expecting the sources in $UMT_PATH". Call script with clone_umt first.
        echo "*************************************************************************************"
        exit 0;
    fi

    pushd $UMT_PATH
    mkdir -p ${UMT_INSTALL_PATH}
    pushd ${UMT_PATH}/umt_workspace
    echo "*** Fetch and build dependencies ***"

    git clone https://github.com/LLNL/UMT_TPLS.git

    tar xvzf UMT_TPLS/metis-5.1.0.tar.gz
    pushd metis-5.1.0
    cmake . -DCMAKE_INSTALL_PREFIX=${UMT_INSTALL_PATH} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DGKLIB_PATH=${PWD}/GKlib
    make -j install
    popd

    tar xvzf UMT_TPLS/conduit-v0.8.8-src-with-blt.tar.gz
    mkdir build_conduit
    pushd build_conduit
    cmake ${PWD}/../conduit-v0.8.8/src -DCMAKE_INSTALL_PREFIX=${UMT_INSTALL_PATH} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_Fortran_COMPILER=${FC} -DMPI_CXX_COMPILER=${MPI_PATH}/bin/mpicxx -DMPI_Fortran_COMPILER=${MPI_PATH}/bin/mpifort -DBUILD_SHARED_LIBS=OFF -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_DOCS=OFF -DENABLE_FORTRAN=ON -DENABLE_MPI=ON -DENABLE_PYTHON=OFF
    make -j install
    popd

    tar xvzf UMT_TPLS/v2.24.0.tar.gz
    mkdir build_hypre
    pushd build_hypre
    cmake ${PWD}/../hypre-2.24.0/src -DCMAKE_INSTALL_PREFIX=${UMT_INSTALL_PATH} -DCMAKE_C_COMPILER=${CC}
    make -j install
    popd

    unzip -o UMT_TPLS/v4.4.zip
    mkdir build_mfem
    pushd build_mfem
    cmake ${PWD}/../mfem-4.4 -DCMAKE_INSTALL_PREFIX=${UMT_INSTALL_PATH} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DMPI_CXX_COMPILER=${MPI_PATH}/bin/mpicxx -DMFEM_USE_MPI=TRUE -DMFEM_USE_CONDUIT=TRUE -DMFEM_USE_METIS_5=TRUE -DCMAKE_PREFIX_PATH=${INSTALL_PATH}}
    make -j install
    popd

    # apply patch
    pushd ${UMT_PATH}
    git apply ${UMT_PATCH}/umt-git.patch
    popd

    # Build UMT
    mkdir build_umt
    #pushd build_umt

    # Run CMake on UMT, compile, and install.
    cmake ${UMT_PATH}/src \
                    -DCMAKE_BUILD_TYPE=Release \
    			    -DSTRICT_FPP_MODE=YES \
                    -DENABLE_OPENMP=YES \
    			    -DENABLE_OPENMP_OFFLOAD=Yes \
                    -DOPENMP_HAS_USE_DEVICE_ADDR=Yes \
    			    -DOPENMP_HAS_FORTRAN_INTERFACE=NO \
                    -DCMAKE_CXX_COMPILER=${CXX} \
    			    -DCMAKE_Fortran_COMPILER=${FC} \
                    -DCMAKE_INSTALL_PREFIX=${UMT_INSTALL_PATH} \
    			    -DMETIS_ROOT=${UMT_INSTALL_PATH} \
                    -DHYPRE_ROOT=${UMT_INSTALL_PATH} \
    			    -DMFEM_ROOT=${UMT_INSTALL_PATH} \
                    -DCONDUIT_ROOT=${UMT_INSTALL_PATH} $1
    make -j install
    popd

    # undo patch
    pushd ${UMT_PATH}
    git apply -R ${UMT_PATCH}/umt-git.patch
    popd

popd
exit 1
fi

# Run UMT
if [ "$1" == "run_umt" ]; then
    ${UMT_INSTALL_PATH}/bin/makeUnstructuredBox
    ${UMT_INSTALL_PATH}/bin/test_driver -i ./unstructBox3D.mesh -r 1 -R 6 -b 1
    exit 1
fi

usage