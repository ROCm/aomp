#!/bin/bash
#
#  tr_build_aomp.sh : Build aomp using TheRock 

_workdir=${1:-/work}
_aomprepodir=$_workdir/$USER/git/tr_aomp
_therockdir=$_aomprepodir/TheRock
_curdir=$PWD

cd $_therockdir

_config_out=config.out
_build_out=build.out
_build_dist_out=$PWD/build_dist.out
_setup_ccache_out=setup_ccache.out

[ -d build ] && rm -rf build
[ -f $_setup_ccache_out ] && rm $_setup_ccache_out
[ -f $_build_out ] && rm  $_build_out
[ -f $_config_out ] && rm $_config_out
[ -f $_build_dist_out ] && rm $_build_dist_out

#_cmd="cmake -B build -GNinja -DTHEROCK_AMDGPU_TARGETS=gfx90a -DTHEROCK_ENABLE_COMPOSABLE_KERNEL=OFF -DTHEROCK_ENABLE_FFT=OFF -DTHEROCK_ENABLE_RAND=OFF -DTHEROCK_ENABLE_PRIM=OFF -DTHEROCK_ENABLE_BLAS=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache $PWD"
#_cmd="cmake -B build -GNinja -DTHEROCK_AMDGPU_TARGETS=gfx90a -DTHEROCK_ENABLE_COMPOSABLE_KERNEL=OFF -DTHEROCK_ENABLE_MATH_LIBS=OFF -DTHEROCK_ENABLE_ML_LIBS=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DTHEROCK_BUNDLE_SYSDEPS=OFF $PWD"
#
_cmd="cmake -B build -GNinja -DTHEROCK_AMDGPU_TARGETS=gfx90a -DTHEROCK_ENABLE_COMPOSABLE_KERNEL=OFF -DTHEROCK_ENABLE_MATH_LIBS=OFF -DTHEROCK_ENABLE_ML_LIBS=OFF -DTHEROCK_BUNDLE_SYSDEPS=OFF $PWD"
eval "$(python3 ./build_tools/setup_ccache.py)" 2>&1 >>$_setup_ccache_out
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
pwd >> $_build_dist_out
echo "===> CMD:$_cmd" >> $_build_dist_out
date >> $_build_dist_out
$_cmd 2>&1 >> $_build_dist_out
[ $? != 0 ] && cd $_curdir && exit 1 
date >> $_build_dist_out
echo "DONE see $_config_out and $_build_out and $_build_dist_out " | tee -a  $_build_dist_out
