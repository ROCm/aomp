# ROCM_AOMP_SOURCE_INSTALL
## Setup
Choose directory for build, ~/ROCm is what is shown here.
```
mkdir -p ~/ROCm/src; mkdir -p ~/ROCm/out
cd ~/ROCm/src
```
**Setup Build Environment**
```
export OUT_DIR=~/ROCm/out \
export SRC_DIR=~/ROCm/src/aomp \
export ROCM_RPATH='$ORIGIN':'$ORIGIN/../lib':'$ORIGIN/../../lib':'$ORIGIN/../hsa/lib':'$ORIGIN/../../hsa/lib'

export AOMP=$OUT_DIR/aomp \
export AOMP_REPOS=$SRC_DIR \
export AOMP_STANDALONE_BUILD=0
```
**Download necessary repositories.**
```
repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-3.9.x
repo sync aomp/roct-thunk-interface \
aomp/rocr-runtime \
aomp/rocm-device-libs \
aomp/rocm-compilersupport \
aomp/hip-on-vdi \
aomp/aomp \
aomp/aomp-extras \
aomp/flang \
aomp/amd-llvm-project \
aomp/vdi \
aomp/opencl-on-vdi
llvm-project
```
```
mv llvm_amd-stg-open $SRC_DIR
mkdir -p aomp/build; cd aomp/build
mkdir roct; mkdir llvm-project; mkdir device-libs; mkdir rocr; mkdir comgr; mkdir rocclr; mkdir hip-on-vdi; mkdir rocminfo
```
## Build
#### ROCt
```
cd $SRC_DIR/build/roct
cmake \
-DCMAKE_MODULE_PATH="$AOMP_REPOS/roct-thunk-interface/cmake_modules" \
-DCMAKE_BUILD_TYPE="Release" \
-DBUILD_SHARED_LIBS="ON" \
-DCMAKE_PREFIX_PATH="$OUT_DIR" \
-DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH="FALSE" \
-DCMAKE_INSTALL_RPATH="$OUT_DIR" \
-DHSAKMT_WERROR=1 \
"$SRC_DIR/roct-thunk-interface"

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
"$SRC_DIR/llvm_amd-stg-open/llvm"

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
-DCMAKE_INSTALL_RPATH="$ROCM_RPATH" \
"$SRC_DIR/rocm-device-libs"

make -j<num_jobs>; make install
```
#### ROCr
```
cd $SRC_DIR/build/rocr
cmake \
-DCMAKE_BUILD_TYPE="Release" \
-DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
-DCMAKE_PREFIX_PATH="$SRC_DIR/build/device-libs;$OUT_DIR;" \
"$SRC_DIR/rocr-runtime/src"

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
"$SRC_DIR/rocm-compilersupport/lib/comgr"

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
-DOPENCL_DIR="$SRC_DIR/opencl-on-vdi" \
-DCMAKE_PREFIX_PATH="$OUT_DIR;$OUT_DIR/llvm" \
-DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
"$SRC_DIR/vdi"

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
-DCPACK_INSTALL_PREFIX="$OUT_DIR/hip" \
-DCMAKE_SHARED_LINKER_FLAGS=" -Wl,--enable-new-dtags -Wl,--rpath,${ROCM_RPATH}"\
-DROCM_PATH=$OUT_DIR \
-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE\
-DCMAKE_SKIP_BUILD_RPATH=TRUE \
"$SRC_DIR/hip-on-vdi"

make -j<num_jobs>; make install
```
#### ROCdbapi
```
cd $SRC_DIR/build/rocdbgapi
cmake \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=$OUT_DIR \
-DCPACK_PACKAGING_INSTALL_PREFIX="$OUT_DIR" \
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE \
-DCMAKE_INSTALL_RPATH=$ROCM_RPATH \
$SRC_DIR/ROCdbgapi

```
#### ROCgdb
```
cd $SRC_DIR/build/rocgdb
$SRC_DIR/ROCgdb/configure --program-prefix=roc --prefix="$OUT_DIR" --enable-64-bit-bfd \
    --with-pkgversion="3.9.0" \
    --enable-targets="x86_64-linux-gnu,amdgcn-amd-amdhsa" \
    --disable-ld --disable-gas --disable-gdbserver --disable-sim --enable-tui \
    --disable-gdbtk --disable-shared  \
    --with-expat --with-system-zlib --without-guile --with-babeltrace --with-lzma \
    --with-python=python3 --with-rocm-dbgapi=$OUT_DIR

LD_RUN_PATH='${ORIGIN}/../lib' make -j<num_jobs> all-gdb
make install-strip-gdb

```
#### AOMP
```
cd $SRC_DIR/aomp/bin
ROCM_DIR=$OUT_DIR AOMP_APPLY_ROCM_PATCHES=0 AOMP_CHECK_GIT_BRANCH=0 AOMP_BUILD_DEBUG=1 ./build_aomp.sh
```
### Notes
- If installation is not into /opt/rocm, add --rocm-path=$OUT_DIR to command line.
- export LD_LIBRARY_PATH=$OUT_DIR/lib
