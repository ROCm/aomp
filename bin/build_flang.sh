#!/bin/bash
# 
#  build_flang.sh:  Script to build the flang component of the AOMP compiler. 
#
#
BUILD_TYPE=${BUILD_TYPE:-Release}

# --- Start standard header ----
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
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_FLANG=${INSTALL_FLANG:-$AOMP_INSTALL_DIR}

if [ "$AOMP_PROC" == "ppc64le" ] ; then
   TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}PowerPC"
else
   if [ "$AOMP_PROC" == "aarch64" ] ; then
      TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}AArch64"
   else
      TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}X86"
   fi
fi

#MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_FLANG -DLLVM_ENABLE_ASSERTIONS=ON $AOMP_ORIGIN_RPATH -DLLVM_CONFIG=$INSTALL_FLANG/bin/llvm-config -DCMAKE_CXX_COMPILER=$AOMP_INSTALL_DIR/bin/clang++ -DCMAKE_C_COMPILER=$AOMP_INSTALL_DIR/bin/clang -DCMAKE_Fortran_COMPILER=gfortran -DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD -DFLANG_OPENMP_GPU_AMD=ON -DFLANG_OPENMP_GPU_NVIDIA=ON -DLLVM_INSTALL_TOOLCHAIN_ONLY=ON"

MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE \
-DCMAKE_INSTALL_PREFIX=$INSTALL_FLANG \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_CONFIG=$OUT_DIR/llvm/bin/llvm-config \
-DCMAKE_CXX_COMPILER=$OUT_DIR/llvm/bin/clang++ \
-DCMAKE_C_COMPILER=$OUT_DIR/llvm/bin/clang \
-DCMAKE_Fortran_COMPILER=gfortran \
-DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD \
-DFLANG_OPENMP_GPU_AMD=ON \
-DFLANG_OPENMP_GPU_NVIDIA=ON \
-DLLVM_INSTALL_TOOLCHAIN_ONLY=ON"


if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  help_build_aomp
fi

REPO_BRANCH=$AOMP_FLANG_REPO_BRANCH
REPO_DIR=$AOMP_REPOS/$AOMP_FLANG_REPO_NAME
checkrepo

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_FLANG
   $SUDO touch $INSTALL_FLANG/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_FLANG"
      exit 1
   fi
   $SUDO rm $INSTALL_FLANG/testfile
fi

# Skip synchronization from git repos if nocmake or install are specified
if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/$AOMP_FLANG_REPO_NAME"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_DIR/build/$AOMP_FLANG_REPO_NAME
   mkdir -p $BUILD_DIR/build/$AOMP_FLANG_REPO_NAME

   if [ $COPYSOURCE ] ; then 
      #  Copy/rsync the git repos into /tmp for faster compilation
      mkdir -p $BUILD_DIR
      echo
      echo "WARNING!  BUILD_DIR($BUILD_DIR) != AOMP_REPOS($AOMP_REPOS)"
      echo "SO RSYNCING AOMP_REPOS TO: $BUILD_DIR"
      echo
      echo rsync -a --exclude ".git" --delete $AOMP_REPOS/$AOMP_FLANG_REPO_NAME $BUILD_DIR
      rsync -a --exclude ".git" --delete $AOMP_REPOS/$AOMP_FLANG_REPO_NAME $BUILD_DIR 2>&1
   fi
else
   if [ ! -d $BUILD_DIR/build/$AOMP_FLANG_REPO_NAME ] ; then 
      echo "ERROR: The build directory $BUILD_DIR/build/$AOMP_FLANG_REPO_NAME does not exist"
      echo "       run $0 without nocmake or install options. " 
      exit 1
   fi
fi

cd $BUILD_DIR/build/$AOMP_FLANG_REPO_NAME

#  Need llvm-config to come from previous LLVM build
export PATH=$AOMP_INSTALL_DIR/bin:$PATH

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake ---- " 
   if [ $COPYSOURCE ] ; then
      echo ${AOMP_CMAKE} $MYCMAKEOPTS  $BUILD_DIR/$AOMP_FLANG_REPO_NAME
      ${AOMP_CMAKE} $MYCMAKEOPTS  $BUILD_DIR/$AOMP_FLANG_REPO_NAME 2>&1
   else
      echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_FLANG_REPO_NAME
      ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_FLANG_REPO_NAME 2>&1
   fi
   if [ $? != 0 ] ; then 
      echo "ERROR cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi
fi

echo
echo " -----Running make ---- " 
echo make -j $NUM_THREADS 
make -j $NUM_THREADS 
if [ $? != 0 ] ; then 
   echo "ERROR make -j $NUM_THREADS failed"
   exit 1
fi

if [ "$1" == "install" ] ; then
   echo " -----Installing to $INSTALL_FLANG ---- " 
   $SUDO make install 
   if [ $? != 0 ] ; then 
      echo "ERROR make install failed "
      exit 1
   fi
   $SUDO make install/local
   if [ $? != 0 ] ; then 
      echo "ERROR make install/local failed "
      exit 1
   fi
   echo "SUCCESSFUL INSTALL to $INSTALL_FLANG "
   echo
   if [ -d $OUT_DIR/openmp-extras ]; then
     echo "Add flang symbolic link."
     echo "Copy flang, flang1, flang2 into $OUT_DIR/llvm/bin"
     cp $OUT_DIR/openmp-extras/bin/flang1 $OUT_DIR/llvm/bin
     cp $OUT_DIR/openmp-extras/bin/flang2 $OUT_DIR/llvm/bin
   fi
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo 
fi
