# OPENMPEXTRAS_SOURCE_INSTALL
## Setup
Choose directory for build, ~/ROCm is what is shown here.
```
mkdir -p ~/ROCm/src; mkdir -p ~/ROCm/out
cd ~/ROCm/src
```
```
mkdir -p build; cd build
mkdir roct; mkdir llvm-project; mkdir device-libs; mkdir rocr; mkdir comgr; mkdir rocminfo; mkdir rocclr; mkdir hip-on-vdi; mkdir rocdbgapi; mkdir rocgdb
```
**Setup ROCm Build Environment**
```
export OUT_DIR=~/ROCm/out \
export SRC_DIR=~/ROCm/src \
export ROCM_RPATH='$ORIGIN':'$ORIGIN/../lib':'$ORIGIN/../../lib':'$ORIGIN/../hsa/lib':'$ORIGIN/../../hsa/lib'
```
**Download necessary repositories.**
```
repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-3.9.x
repo sync llvm-project \
ROCT-Thunk-Interface \
ROCR-Runtime \
ROCm-Device-Libs \
ROCm-CompilerSupport \
ROCclr \
HIP \
ROCm-OpenCL-Runtime \
ROCdbgapi \
ROCgdb \
openmp-extras/aomp \
openmp-extras/aomp-extras \
openmp-extras/flang 

mv llvm_amd-stg-open llvm-project
```
## Build
#### ROCt
```
cd $SRC_DIR/build/roct

cmake \
-DCMAKE_MODULE_PATH="$SRC_DIR/ROCT-Thunk-Interface/cmake_modules" \
-DCMAKE_BUILD_TYPE="Release" \
-DBUILD_SHARED_LIBS="ON" \
-DCMAKE_PREFIX_PATH="$OUT_DIR" \
-DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH="FALSE" \
-DCMAKE_INSTALL_RPATH="$OUT_DIR" \
-DHSAKMT_WERROR=1 \
"$SRC_DIR/ROCT-Thunk-Interface"

make -j<num_jobs>; make install
```
#### ROCm LLVM
```
cd $SRC_DIR/build/llvm-project

cmake -DCMAKE_BUILD_TYPE="Release" \
-DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
-DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra;compiler-rt" \
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH="FALSE" \
-DCMAKE_INSTALL_RPATH="$OUT_DIR" \
-DCMAKE_INSTALL_PREFIX="$OUT_DIR/llvm" \
-DLLVM_ENABLE_ASSERTIONS="$ENABLE_ASSERTIONS" \
-DLLVM_ENABLE_Z3_SOLVER=OFF \
-DLLVM_ENABLE_ZLIB=OFF \
"$SRC_DIR/llvm-project/llvm"

make -j<num_jobs>; make install
```
#### Device Libs
```
cd $SRC_DIR/build/device-libs
cmake \
-DCMAKE_BUILD_TYPE="Release" \
-DCMAKE_PREFIX_PATH="$SRC_DIR/build/llvm-project" \
-DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH="FALSE" \
"$SRC_DIR/ROCm-Device-Libs"

make -j<num_jobs>; make install
```
#### ROCr
```
cd $SRC_DIR/build/rocr
cmake \
-DCMAKE_BUILD_TYPE="Release" \
-DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
-DCMAKE_PREFIX_PATH="$SRC_DIR/build/device-libs;$OUT_DIR;" \
"$SRC_DIR/ROCR-Runtime/src"

make -j<num_jobs>; make install
```
#### Comgr
```
cd $SRC_DIR/build/comgr
cmake \
-DCMAKE_PREFIX_PATH="$SRC_DIR/build/llvm-project;$SRC_DIR/build/device-libs;$OUT_DIR" \
-DCMAKE_BUILD_TYPE="Release" \
-DBUILD_SHARED_LIBS="ON" \
-DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE \
-DCMAKE_INSTALL_RPATH=$OUT_DIR \
"$SRC_DIR/ROCm-CompilerSupport/lib/comgr"

make -j<num_jobs>; make install
```
#### Rocminfo
```
cd $SRC_DIR/build/rocminfo
cmake \
-DROCRTST_BLD_TYPE="Release" \
-DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
-DCMAKE_PREFIX_PATH="$OUT_DIR" \
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH="FALSE" \
-DCMAKE_SKIP_BUILD_RPATH=TRUE \
-DCMAKE_EXE_LINKER_FLAGS="-Wl,--enable-new-dtags -Wl,--rpath,$ROCM_RPATH" \
"$SRC_DIR/rocminfo"

make -j<num_jobs>; make install
```
#### ROCclr
```
cd $SRC_DIR/build/rocclr
cmake -DUSE_COMGR_LIBRARY="ON" \
-DBUILD_SHARED_LIBS="ON" \
-DCMAKE_BUILD_TYPE="Release" \
-DOPENCL_DIR="$SRC_DIR/ROCm-OpenCL-Runtime" \
-DCMAKE_PREFIX_PATH="$OUT_DIR;$OUT_DIR/llvm" \
-DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
"$SRC_DIR/ROCclr"

make -j<num_jobs>; make install
```
#### HIP
```
cd $SRC_DIR/build/hip-on-vdi
cmake -DCMAKE_BUILD_TYPE=Release \
-DBUILD_SHARED_LIBS=ON \
-DHIP_COMPILER=clang \
-DHIP_PLATFORM=rocclr \
-DCMAKE_PREFIX_PATH="$SRC_DIR/build/rocclr;$OUT_DIR;$OUT_DIR/llvm " \
-DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
-DHSA_PATH="$OUT_DIR/hsa" \
-DCPACK_INSTALL_PREFIX="$OUT_DIR" \
-DCMAKE_SHARED_LINKER_FLAGS=" -Wl,--enable-new-dtags -Wl,--rpath,${ROCM_RPATH}"\
-DROCM_PATH=$OUT_DIR \
-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE\
-DCMAKE_SKIP_BUILD_RPATH=TRUE \
"$SRC_DIR/HIP"

make -j<num_jobs>; make install
```
#### OpenMP-Extras
```
export AOMP=$OUT_DIR/llvm
cd $SRC_DIR/openmp-extras/aomp/bin

BUILD_AOMP=$OUT_DIR/build/openmp-extras \
AOMP_STANDALONE_BUILD=0 \
ROCM_DIR=$OUT_DIR \
HIP_DEVICE_LIB_PATH="$ROCM_DIR/amdgcn/bitcode" \
AOMP_APPLY_ROCM_PATCHES=0 \
AOMP_CHECK_GIT_BRANCH=0 \
AOMP_JENKINS_BUILD_LIST="extras openmp pgmath flang flang_runtime" \
AOMP_REPOS=$SRC_DIR/openmp-extras \
DEVICELIBS_ROOT=$SRC_DIR/ROCm-Device-Libs \
CCC_OVERRIDE_OPTIONS=+--rocm-path=/home/estewart/ROCm/out \
BUILD_AOMP=$SRC_DIR \
./build_aomp.sh
```
### Notes
-If installation is not into /opt/rocm, add --rocm-path=$OUT_DIR to command line.
-Resolve libraries:
```
export LD_LIBRARY_PATH=$OUT_DIR/lib \
LIBRARY_PATH=$OUT_DIR/lib
```

