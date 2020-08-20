#!/bin/bash
# 
#  build_qmcpack.sh: 
#
# --- Start standard header to set build environment variables ----
function getdname(){
   local __DIRN=`dirname "$1"`
   if [ "$__DIRN" = "." ] ; then
      __DIRN=$PWD;
   else
      if [ ${__DIRN:0:1} != "/" ] ; then
         if [ ${__DIRN:0:2} == ".." ] ; then
               __DIRN=`dirname $PWD`/${__DIRN:3}
         else
            if [ ${__DIRN:0:1} = "." ] ; then
               __DIRN=$PWD/${__DIRN:2}
            else
               __DIRN=$PWD/$__DIRN
            fi
         fi
      fi
   fi
   echo $__DIRN
}
thisdir=$(getdname $0)
[ ! -L "$0" ] || thisdir=$(getdname `readlink "$0"`)
. $thisdir/aomp_common_vars
# --- end standard header ----

QMCPACK_REPO=${QMCPACK_REPO:-$AOMP_REPOS_TEST/$AOMP_QMCPACK_REPO_NAME}
build_folder=/tmp/build_MI60_AOMP_offload_real_MP
if [ -d $build_folder ] ; then 
   echo "FRESH START"
   echo rm -rf $build_folder
   rm -rf $build_folder
fi

where_module=`which module`
if [ $? == 0 ] ; then 
   # This script assumes modules setup on poplar cluster
   AOMP="/home/users/coe0046/rocm/aomp_11.7-1"
   module load cmake
   #module load PrgEnv-cray
   module load gcc
   module load hdf5/1.10.1
   module load openblas
   module load rocm-alt
   export FFTW_HOME=/cm/shared/apps/fftw/openmpi/gcc/64/3.3.8
   export BOOST_ROOT=/cm/shared/opt/boost/1.72.0
else
   # Need the following packaages from ubuntu: libxml2-dev, libfftw3-dev, libboost-dev
   export BOOST_ROOT=/usr/lib/x86_64-linux-gnu
   export FFTW_HOME=/usr/lib/x86_64-linux-gnu
fi
 
echo
echo "###################################"
echo Building MI60_AOMP_offload_real_MP
echo "###################################"
thisgpu=`$AOMP/bin/mygpu`
AOMP_GPU=${AOMP_GPU:-$thisgpu}
if [ "$mygpu" == "unknown" ] ; then 
   echo "ERROR: No gpu found"
   exit 1
fi

save_current_dir=$PWD
mkdir -p $build_folder
cd $build_folder
 
MYCMAKEOPTS="-DCMAKE_C_COMPILER=$AOMP/bin/clang -DCMAKE_CXX_COMPILER=$AOMP/bin/clang++ \
      -DQMC_MPI=0 \
      -DCMAKE_C_FLAGS="""-march=native""" \
      -DCMAKE_CXX_FLAGS="""-march=native -Xopenmp-target=amdgcn-amd-amdhsa -march=$mygpu"""\
      -DQMC_MIXED_PRECISION=1 -DENABLE_OFFLOAD=ON -DOFFLOAD_TARGET="""amdgcn-amd-amdhsa""" \
      -DENABLE_TIMERS=1 "
echo $AOMP_CMAKE $MYCMAKEOPTS $QMCPACK_REPO
$AOMP_CMAKE $MYCMAKEOPTS $QMCPACK_REPO
echo
echo make -j$NUM_THREADS
make -j$NUM_THREADS

cd $save_current_dir
echo 
echo "DONE!  Build is in $build_folder. To test:"
echo "       cd $build_folder"
echo "       ctest -R deterministic"
echo 
