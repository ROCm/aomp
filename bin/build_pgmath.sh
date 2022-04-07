#!/bin/bash
# 
#  build_pgmath.sh:  Script to build the pgmath component of the AOMP compiler. 
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

COMP_INC_DIR=$(ls -d $AOMP_INSTALL_DIR/lib/clang/*/include )

if [ "$AOMP_PROC" == "ppc64le" ] ; then
    MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_FLANG -DLLVM_ENABLE_ASSERTIONS=ON $AOMP_ORIGIN_RPATH -DCMAKE_Fortran_COMPILER=$AOMP_INSTALL_DIR/bin/flang -DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD -DCMAKE_C_FLAGS=-I$COMP_INC_DIR -DCMAKE_CXX_FLAGS=-I$COMP_INC_DIR"
else
    MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_FLANG -DLLVM_ENABLE_ASSERTIONS=ON $AOMP_ORIGIN_RPATH -DLLVM_CONFIG=$INSTALL_FLANG/bin/llvm-config -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang  -DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD -DCMAKE_C_FLAGS=-I$COMP_INC_DIR -DCMAKE_CXX_FLAGS=-I$COMP_INC_DIR"
fi

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  help_build_aomp
fi

REPO_DIR=$AOMP_REPOS/$AOMP_FLANG_REPO_NAME

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
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/pgmath"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_DIR/build/pgmath
   mkdir -p $BUILD_DIR/build/pgmath
else
   if [ ! -d $BUILD_DIR/build/pgmath ] ; then 
      echo "ERROR: The build directory $BUILD_DIR/build/pgmath does not exist"
      echo "       run $0 without nocmake or install options. " 
      exit 1
   fi
fi

cd $BUILD_DIR/build/pgmath 

#  Need llvm-config to come from previous LLVM build
export PATH=$AOMP_INSTALL_DIR/bin:$PATH

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/runtime/libpgmath
   ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/runtime/libpgmath  2>&1
   if [ $? != 0 ] ; then 
      echo "ERROR cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi
fi

echo
echo " -----Running make ---- " 
echo make -j $AOMP_JOB_THREADS 
make -j $AOMP_JOB_THREADS 
if [ $? != 0 ] ; then 
   echo "ERROR make -j $AOMP_JOB_THREADS failed"
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
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo 
fi
