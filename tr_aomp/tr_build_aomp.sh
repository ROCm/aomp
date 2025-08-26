#!/bin/bash
#
#  tr_build_aomp.sh : Build aomp using TheRock 
#
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/tr_aomp_common_vars"
# --- end standard header ----
#
if [ -z $AOMP_INSTALL_DIR ] ; then
   echo "ERROR: Env VAR AOMP_INSTALL_DIR is not set "
   cd $_curdir
   exit 1
fi

_therockdir=$TR_AOMP_REPOS/TheRock
_curdir=$PWD

cd $_therockdir

_config_out=$_therockdir/build/config.out
_build_out=$_therockdir/build/build.out
_dist_out=$_therockdir/build/build_dist.out
_setup_ccache_out=$_therockdir/build/setup_ccache.out

[ -d build ] && rm -rf build
mkdir -p $_therockdir/build
[ -f $_setup_ccache_out ] && rm $_setup_ccache_out
[ -f $_build_out ] && rm  $_build_out
[ -f $_config_out ] && rm $_config_out
[ -f $_dist_out ] && rm $_dist_out

if [ ${AOMP_SKIP_RCCL} == 1 ] ; then
   _rccl_opt="-DTHEROCK_ENABLE_RCCL=OFF"
else
   _rccl_opt="-DTHEROCK_ENABLE_RCCL=ON"
fi
if [ ${AOMP_SKIP_MATH_LIBS} == 1 ] ; then
   _mathlibs_opt="-DTHEROCK_ENABLE_MATH_LIBS=OFF"
else
   _mathlibs_opt="-DTHEROCK_ENABLE_MATH_LIBS=ON"
fi

_cmd="cmake -B build -GNinja -DTHEROCK_AMDGPU_TARGETS=gfx90a -DTHEROCK_ENABLE_COMPOSABLE_KERNEL=OFF $_mathlibs_opt  -DTHEROCK_ENABLE_ML_LIBS=OFF -DTHEROCK_BUNDLE_SYSDEPS=ON -DTHEROCK_BUILD_TESTING=OFF $_rccl_opt $_therockdir"
$thisdir/tr_add_info therock_config $_cmd
$thisdir/tr_add_info build_path $PATH

eval "$(python3 ./build_tools/setup_ccache.py)" 2>&1 >>$_setup_ccache_out
echo "===> CMAKE CMD:$_cmd"
echo "===> CMD:$_cmd" >> $_config_out
date >> $_config_out
$_cmd 2>&1 >> $_config_out
[ $? != 0 ] && cd $_curdir && exit 1 

_cmd="cmake --build build"
echo "===> CMD:$_cmd" >>  $_build_out
date >> $_build_out
$_cmd 2>&1 >> $_build_out
[ $? != 0 ] && cd $_curdir && exit 1 
date >> $_build_out

cd build
_cmd="ninja therock-dist"
pwd >> $_dist_out
echo "===> CMD:$_cmd" >> $_dist_out
date >> $_dist_out
$_cmd 2>&1 >> $_dist_out
[ $? != 0 ] && cd $_curdir && exit 1 
date >> $_dist_out

if [ -z $AOMP_INSTALL_DIR ] ; then
   echo "ERROR: Env VAR AOMP_INSTALL_DIR is not set "
   cd $_curdir
   exit 1
else
   if [ -d $AOMP_INSTALL_DIR ] ; then
      rm -rf $AOMP_INSTALL_DIR
   fi
fi
echo mkdir -p $AOMP_INSTALL_DIR
mkdir -p $AOMP_INSTALL_DIR
echo rsync -a dist/rocm/ $AOMP_INSTALL_DIR/
rsync -a dist/rocm/ $AOMP_INSTALL_DIR/
echo ln -sf $AOMP_INSTALL_DIR $AOMP
ln -sf $AOMP_INSTALL_DIR $AOMP

# Convenience link for rebuilding the compiler.
ln -sf $TR_AOMP_REPOS/TheRock/build/compiler/amd-llvm/build $TR_AOMP_REPOS/llvm-project/build
ln -sf $TR_AOMP_REPOS/aomp/tr_aomp/build_install_aomp_from_therock.sh $TR_AOMP_REPOS/llvm-project/build/.

_date=`date`
$thisdir/tr_add_info build_date $_date
echo "DONE see $_config_out and $_build_out and $_dist_out " | tee -a  $_dist_out


