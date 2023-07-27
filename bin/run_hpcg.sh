#! /usr/bin/env bash

# HPCG_SOURCE_DIR           where to clone sources to. Default: AOMP_REPOS_TEST
# HPCG_BUILD_NUM_THREADS    Number of parallel compile processes

realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars

: ${HPCG_SOURCE_DIR:=$AOMP_REPOS_TEST/hpcg}
: ${HPCG_BUILD_NUM_THREADS:=32}

rm -rf ${HPCG_SOURCE_DIR}

if [ ! -d ${HPCG_SOURCE_DIR} ]; then
  mkdir -p ${HPCG_SOURCE_DIR}
fi

cd ${HPCG_SOURCE_DIR} || exit 1

if [ ! -d ./rocHPCG ]; then
  # Get the sources
  git clone --depth 1 --single-branch -b omptarget https://github.com/ROCmSoftwarePlatform/rocHPCG || exit 1
  cd rocHPCG || exit 1
  git checkout omptarget || exit 1
fi

cd omptarget/hpcg || exit 1

mkdir build
cd build || exit 1

../configure LLVM_OMP_TARGET

make -j${HPCG_BUILD_NUM_THREADS}

cd bin
./xhpcg 104 104 104 600

cat *.txt | grep -e "INVALID"
if [ $? -eq 1 ]; then
  echo "Invalid result produced, exiting."
  exit 1
fi

