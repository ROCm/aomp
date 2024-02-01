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

COMP_INC_DIR=$(ls -d $INSTALL_PREFIX/llvm/lib/clang/*/include )

if [ "$AOMP_PROC" == "ppc64le" ] ; then
    MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_FLANG -DLLVM_ENABLE_ASSERTIONS=ON $AOMP_ORIGIN_RPATH -DCMAKE_Fortran_COMPILER=$AOMP_INSTALL_DIR/bin/flang -DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD -DCMAKE_C_FLAGS=-I$COMP_INC_DIR -DCMAKE_CXX_FLAGS=-I$COMP_INC_DIR"
else
    #MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_FLANG -DLLVM_ENABLE_ASSERTIONS=ON $AOMP_ORIGIN_RPATH -DLLVM_CONFIG=$INSTALL_FLANG/bin/llvm-config -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang  -DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD -DCMAKE_C_FLAGS=-I$COMP_INC_DIR -DCMAKE_CXX_FLAGS=-I$COMP_INC_DIR"
    MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_FLANG \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_CONFIG=$INSTALL_FLANG/bin/llvm-config \
    -DCMAKE_CXX_COMPILER=$INSTALL_PREFIX/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=$INSTALL_PREFIX/llvm/bin/clang \
    -DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_FLANG"
fi

if [ "$AOMP_STANDALONE_BUILD" == 0 ]; then
  MYCMAKEOPTS="$MYCMAKEOPTS
  "
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
   ASAN_LIB_PATH=$(${INSTALL_PREFIX}/llvm/bin/clang --print-runtime-dir)
   ASAN_RPATH_FLAGS="-Wl,-rpath=$ASAN_LIB_PATH -L$ASAN_LIB_PATH"
   CXXFLAGS="$CXXFLAGS $ASAN_RPATH_FLAGS -I$COMP_INC_DIR"
   CFLAGS="$CFLAGS $ASAN_RPATH_FLAGS -I$COMP_INC_DIR"
   ASAN_CMAKE_OPTS="$MYCMAKEOPTS -DCMAKE_INSTALL_LIBDIR=lib/asan"
fi

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
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/pgmath"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_DIR/build/pgmath
   mkdir -p $BUILD_DIR/build/pgmath

   if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
      mkdir -p $BUILD_DIR/build/pgmath/asan
   fi

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
   if [ ! -d $BUILD_DIR/build/pgmath ] ; then
      echo "ERROR: The build directory $BUILD_DIR/build/pgmath does not exist"
      echo "       run $0 without nocmake or install options. "
      exit 1
   fi
   if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
      if [ ! -d $BUILD_DIR/build/pgmath/asan ] ; then
         echo "ERROR: The build directory $BUILD_DIR/build/pgmath/asan does not exist"
         echo "       run $0 without nocmake or install options. "
         exit 1
      fi
   fi
fi

#  Need llvm-config to come from previous LLVM build
export PATH=$AOMP_INSTALL_DIR/bin:$PATH

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   if [ "$AOMP_BUILD_SANITIZER" != 1 ];then
      cd $BUILD_DIR/build/pgmath
      echo
      echo " -----Running cmake ---- "
      if [ $COPYSOURCE ] ; then
         echo ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_C_FLAGS=-I$COMP_INC_DIR -DCMAKE_CXX_FLAGS=-I$COMP_INC_DIR $BUILD_DIR/$AOMP_FLANG_REPO_NAME/runtime/libpgmath
         env "$@" ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_C_FLAGS=-I$COMP_INC_DIR -DCMAKE_CXX_FLAGS=-I$COMP_INC_DIR $BUILD_DIR/$AOMP_FLANG_REPO_NAME/runtime/libpgmath  2>&1
      else
         echo ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_C_FLAGS=-I$COMP_INC_DIR -DCMAKE_CXX_FLAGS=-I$COMP_INC_DIR $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/runtime/libpgmath
         env "$@" ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_C_FLAGS=-I$COMP_INC_DIR -DCMAKE_CXX_FLAGS=-I$COMP_INC_DIR $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/runtime/libpgmath  2>&1
      fi
      if [ $? != 0 ] ; then
         echo "ERROR cmake failed. Cmake flags"
         echo "      $MYCMAKEOPTS"
         exit 1
      fi
   fi
   if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
      cd $BUILD_DIR/build/pgmath/asan
      echo
      echo " -----Running cmake ---- "
      if [ $COPYSOURCE ] ; then
         echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$CFLAGS'" -DCMAKE_CXX_FLAGS="'$CXXFLAGS'" $BUILD_DIR/$AOMP_FLANG_REPO_NAME/runtime/libpgmath
         env "$@" ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$CFLAGS'" -DCMAKE_CXX_FLAGS="'$CXXFLAGS'" $BUILD_DIR/$AOMP_FLANG_REPO_NAME/runtime/libpgmath  2>&1
      else
         echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$CFLAGS'" -DCMAKE_CXX_FLAGS="'$CXXFLAGS'" $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/runtime/libpgmath
         env "$@" ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$CFLAGS'" -DCMAKE_CXX_FLAGS="'$CXXFLAGS'" $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/runtime/libpgmath  2>&1
      fi
      if [ $? != 0 ] ; then
         echo "ERROR pgmath-asan cmake failed. Cmake flags"
         echo "      $ASAN_CMAKE_OPTS"
         exit 1
      fi
   fi
fi

if [ "$AOMP_BUILD_SANITIZER" != 1 ];then
   echo
   echo " -----Running make ---- "
   cd $BUILD_DIR/build/pgmath
   echo make -j $NUM_THREADS
   make -j $NUM_THREADS
   if [ $? != 0 ] ; then
      echo "ERROR make -j $NUM_THREADS failed"
      exit 1
   fi
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
   echo
   echo " -----Running make ----"
   cd $BUILD_DIR/build/pgmath/asan
   echo make -j $NUM_THREADS
   make -j $NUM_THREADS
   if [$? != 0 ] ; then
      echo "ERROR pgmath-asan make -j $NUM_THREADS failed"
      exit 1
   fi
fi

if [ "$1" == "install" ] ; then
   if [ "$AOMP_BUILD_SANITIZER" != 1 ];then
      cd $BUILD_DIR/build/pgmath
      echo " -----Installing to $INSTALL_FLANG ---- "
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
      echo "SUCCESSFUL INSTALL to $INSTALL_FLANG "
   fi
   echo
   if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
      cd $BUILD_DIR/build/pgmath/asan
      echo " ------ Installing to $INSTALL_FLANG/lib/asan ---- "
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
   fi
   echo "SUCCESSFUL INSTALL to $INSTALL_FLANG/lib/asan "
   echo
else
   echo
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo
fi
