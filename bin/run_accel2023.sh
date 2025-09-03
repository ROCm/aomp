#! /usr/bin/env bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 

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
  set -x
  #WLOC=http://roclogin.amd.com/SPEC
  WLOC=https://compute-artifactory.amd.com/artifactory/rocm-generic-local/compiler-infra
  wget --timeout 15 --tries=3 -q $WLOC/accel2023-2.0.18.tar.xz
  wget --timeout 15 --tries=3 -q $WLOC/Accel23-scripts.tar
  tar xf accel2023-2.0.18.tar.xz
  tar xvf Accel23-scripts.tar
  set +x
  ./install.sh -f
else
  cd ${ACCEL2023_SOURCE_DIR} || exit 1
fi
export PATH=$AOMP/../bin:$AOMP/../../bin:$PATH
./runOne
#grep ratio= result/*.log
