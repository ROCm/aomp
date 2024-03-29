#!/bin/bash

#  run_nio: download, build, and execute the nio qmcpack test case
#
# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

QMC_DATA=${QMC_DATA:-/tmp/qmcpack_data/tests/performance}
export QMC_DATA

NIODIR=$QMC_DATA/NiO

# Save the current directory name for any exit following a cd command
_curdir=$PWD

if [ ! -d $NIODIR ] ; then 
  _nio_tarball=NiO.tar.gz
  _webserverdir="http://roclogin.amd.com/test"
  mkdir -p $QMC_DATA
  cd $QMC_DATA
  echo
  echo wget $_webserverdir/$_nio_tarball
  wget $_webserverdir/$_nio_tarball >/dev/null
  if [ ! -f $_nio_tarball ] ; then 
    echo "ERROR:  Could not download $_nio_tarball from $_webserverdir"
    echo "        Check your vpn connection!"
    cd $_curdir
    exit 1
  fi
  echo
  echo tar -xvzf $_nio_tarball
  tar -xvzf $_nio_tarball
  echo
  if [ ! -d $NIODIR ] ; then 
    echo "ERROR:  Could untar to NiO directory $NIODIR"
    cd $_curdir
    exit 1
  fi
else
  echo
  echo "WARNING: Using existing NiO directory $NIODIR"
  echo
fi

export OPENMPI_INSTALL=$AOMP_SUPP/openmpi
if [ ! -d $OPENMPI_INSTALL ] ; then 
  echo "ERROR: Missing directory $OPENMP_INSTALL"
  echo "       Consider running 'build_supp.sh openmpi'"
  cd $_curdir
  exit 1
fi
export PATH=$OPENMPI_INSTALL/bin:$PATH

#export CMAKE_PREFIX_PATH=$AOMP/lib/cmake
export CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake
if [ ! -d $CMAKE_PREFIX_PATH ] ; then
  echo "ERROR: Missing directory $CMAKE_PREFIX_PATH"
  echo "       You probally need rocm installed"
  cd $_curdir
  exit 1
fi

setaompgpu
build_folder=build_AOMP_offload_real_MP_$AOMP_GPU
QMCPACK_REPO=${QMCPACK_REPO:-$AOMP_REPOS_TEST/$AOMP_QMCPACK_REPO_NAME}
_build_dir=$QMCPACK_REPO/$build_folder

#  Run S32 as default, allow S8 as alternative if provided on cmdline 
_rundir="$_build_dir/tests/performance/NiO/dmc-a128-e1536-DU32-batched_driver"
_input_file_name="NiO-fcc-S32-dmc.xml"
if [ -n $1 ] ; then
    if [ "$1" == 'S8' ] ; then
        _rundir="$_build_dir/tests/performance/NiO/dmc-a32-e384-DU32-batched_driver"
        _input_file_name="NiO-fcc-S8-dmc.xml"
	echo "Running S8 input"
    fi
fi

if [ ! -d $_rundir ] ; then 
  echo cd $QMC_DATA
  cd $QMC_DATA
  [ -f nohup.out ] && rm nohup.out
  echo USE_HIP_OPENMP=1 $thisdir/build_qmcpack.sh 
  mkdir -p /tmp/$USER
  USE_HIP_OPENMP=1 $thisdir/build_qmcpack.sh 2>&1 >/tmp/$USER/build_qmcpack.log
else
  echo
  echo "WARNING: Using existing run directory $_rundir"
  echo
fi


echo
echo cd $_rundir
cd $_rundir

echo "LIBOMPTARGET_SYNC_COPY_BACK=0 HSA_XNACK=1 OMP_NUM_THREADS=7 mpirun -np 8 --tag-output $AOMP/bin/gpurun ../../../../bin/qmcpack $_input_file_name"
LIBOMPTARGET_SYNC_COPY_BACK=0 HSA_XNACK=1 OMP_NUM_THREADS=7 mpirun -np 8 --tag-output $AOMP/bin/gpurun ../../../../bin/qmcpack $_input_file_name | tee stdout

echo
echo DONE $0
echo See output in $_rundir/stdout
echo

cd $_curdir
