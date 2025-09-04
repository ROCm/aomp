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
if [ -d $_therockdir/.venv/bin ] ; then
   PATH=$_therockdir/.venv/bin:$PATH
   export PATH
fi
(
# reconstruct .amd-llvm.smrev using the current SHA
cd compiler/amd-llvm
smrev="../.amd-llvm.smrev"
head -1 $smrev     >  "${smrev}.new"
git rev-parse HEAD >> "${smrev}.new"
cp "${smrev}.new" $smrev
)

[ -d build ] && rm -rf build
mkdir -p $_therockdir/build

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

# Specify target GFXLIST
# gfx not currently supported by TheRock:
#   gfx900 gfx902 gfx90c gfx1031 gfx1052 gfx1053
#   gfx9-generic gfx9-4-generic gfx10-1-generic gfx10-3-generic gfx11-generic gfx12-generic
GFXLIST=${GFXLIST:-"gfx906 gfx908 gfx90a gfx942 gfx950 gfx1010 gfx1011 gfx1012 gfx1030 gfx1032 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1200 gfx1201"}

_gfxsemicolons=$(echo "$GFXLIST" | tr ' ' ';')
_cmd="cmake -B build -GNinja -DTHEROCK_AMDGPU_TARGETS='$_gfxsemicolons' -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=gfx9 -DTHEROCK_ENABLE_COMPOSABLE_KERNEL=OFF $_mathlibs_opt  -DTHEROCK_ENABLE_ML_LIBS=OFF -DTHEROCK_BUNDLE_SYSDEPS=ON -DTHEROCK_BUILD_TESTING=OFF $_rccl_opt $_therockdir"

#Record config and PATH in AOMP release info file"
$thisdir/tr_add_info.sh therock_config $_cmd
$thisdir/tr_add_info.sh build_path $PATH

eval "$(python3 ./build_tools/setup_ccache.py)"
echo 
echo "===== CMD:$_cmd"
date
$_cmd 2>&1
[ $? != 0 ] && cd $_curdir && exit 1 

_cmd="cmake --build build"
echo 
echo "===== CMD:$_cmd"
date
$_cmd 2>&1
[ $? != 0 ] && cd $_curdir && exit 1 
date

cd build
_cmd="ninja therock-dist"
echo 
echo "===== CMD:$_cmd"
date
$_cmd
[ $? != 0 ] && cd $_curdir && exit 1 
date

if [ -z $AOMP_INSTALL_DIR ] ; then
   echo "ERROR: Env variable AOMP_INSTALL_DIR is not set."
   cd $_curdir
   exit 1
else
   if [ -d $AOMP_INSTALL_DIR ] ; then
      rm -rf $AOMP_INSTALL_DIR
   fi
fi
echo
echo "===== copying ROCm build from $_therockdir/build/dest/rocm to $AOMP_INSTALL_DIR" 
echo mkdir -p $AOMP_INSTALL_DIR
mkdir -p $AOMP_INSTALL_DIR
echo rsync -a dist/rocm/ $AOMP_INSTALL_DIR/
rsync -a dist/rocm/ $AOMP_INSTALL_DIR/
echo ln -sf $AOMP_INSTALL_DIR $AOMP
ln -sf $AOMP_INSTALL_DIR $AOMP

# Convenience link for rebuilding the compiler.
echo ln -sr $TR_AOMP_REPOS/TheRock/build/compiler/amd-llvm/build $TR_AOMP_REPOS/llvm-project/build
ln -sr $TR_AOMP_REPOS/TheRock/build/compiler/amd-llvm/build $TR_AOMP_REPOS/llvm-project/build
echo ln -sr $TR_AOMP_REPOS/aomp/tr_aomp/build_install_aomp_from_therock.sh $TR_AOMP_REPOS/llvm-project/build/.
ln -sr $TR_AOMP_REPOS/aomp/tr_aomp/build_install_aomp_from_therock.sh $TR_AOMP_REPOS/llvm-project/build/.

_date=`date`
$thisdir/tr_add_info.sh build_date $_date
[ -d $AOMP_BUILD_LOGS ] && rm -rf $AOMP_BUILD_LOGS
rsync -a $_therockdir/build/logs/ $AOMP_BUILD_LOGS/

echo
echo "===== DONE $0 for AOMP release $AOMP_VERSION_STRING"
echo
