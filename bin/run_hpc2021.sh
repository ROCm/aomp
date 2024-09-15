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

if [ "$1" == "-clean" ]; then
  rm -rf ${HPC2021_SOURCE_DIR}
  mkdir -p ${HPC2021_SOURCE_DIR}
  cd ${HPC2021_SOURCE_DIR} || exit 1
  wget -q http://roclogin.amd.com/SPEC/hpc2021-1.1.9.tar.xz
  wget -q http://roclogin.amd.com/SPEC/Hpc21-scripts.tar
  tar xf hpc2021-1.1.9.tar.xz
  ./install.sh -f
  tar xvf Hpc21-scripts.tar
else
  cd ${HPC2021_SOURCE_DIR} || exit 1
fi
export PATH=$AOMP/../bin:$AOMP/../../bin:$PATH
./runOne
#grep ratio= result/*.log
