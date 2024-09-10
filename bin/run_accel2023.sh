#! /usr/bin/env bash

# ACCEL2023_SOURCE_DIR           where to clone sources to. Default: AOMP_REPOS_TEST
# ACCEL2023_BUILD_NUM_THREADS    Number of parallel compile processes. Default: 32
# export ACC_INPUT=ref   if you wantto run reference isntead of test

export HPG_INPUT=${HPG_INPUT:-test}
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars

: ${ACCEL2023_SOURCE_DIR:=$AOMP_REPOS_TEST/accel2023-2.0.18}
: ${ACCEL2023_BUILD_NUM_THREADS:=32}

if [ "$1" == "-clean" ]; then
  rm -rf ${ACCEL2023_SOURCE_DIR}
  mkdir -p ${ACCEL2023_SOURCE_DIR}
  cd ${ACCEL2023_SOURCE_DIR} || exit 1
  wget http://roclogin.amd.com/SPEC/accel2023-2.0.18.tar.xz
  wget http://roclogin.amd.com/SPEC/Accel23-scripts.tar
  tar xf accel2023-2.0.18.tar.xz
  ./install.sh -f
  tar xvf Accel23-scripts.tar
else
  cd ${ACCEL2023_SOURCE_DIR} || exit 1
fi

./runOne
grep ratio= result/*.log
