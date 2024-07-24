#!/bin/bash
#
#  build_pgmath.sh:  Script to build the pgmath component of the AOMP compiler.
#
#
BUILD_TYPE=${BUILD_TYPE:-Release}

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
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
    MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_FLANG -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_Fortran_COMPILER=$LLVM_INSTALL_LOC/bin/flang -DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD "
else
    MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_FLANG -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_CONFIG=$LLVM_INSTALL_LOC/bin/llvm-config -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang  -DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD "
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
  ASAN_FLAGS="$ASAN_FLAGS -I$COMP_INC_DIR"
  ASAN_CMAKE_OPTS="$MYCMAKEOPTS -DCMAKE_PREFIX_PATH=$AOMP/lib/asan/cmake -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF -DCMAKE_INSTALL_BINDIR=bin/asan -DCMAKE_INSTALL_LIBDIR=lib/asan"
  if [ "$AOMP_STANDALONE_BUILD" == 1 ]; then
    ASAN_CMAKE_OPTS="$ASAN_CMAKE_OPTS $AOMP_ASAN_ORIGIN_RPATH"
  else
    ASAN_CMAKE_OPTS="$ASAN_CMAKE_OPTS $OPENMP_EXTRAS_ORIGIN_RPATH"
  fi
fi

if [ "$AOMP_STANDALONE_BUILD" == 1 ]; then
  MYCMAKEOPTS="$MYCMAKEOPTS $AOMP_ORIGIN_RPATH"
else
  MYCMAKEOPTS="$MYCMAKEOPTS $OPENMP_EXTRAS_ORIGIN_RPATH"
fi

MYCMAKEOPTS="$MYCMAKEOPTS -DCMAKE_PREFIX_PATH=$AOMP/lib/cmake"

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
   if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
      mkdir -p $BUILD_DIR/build/pgmath/asan
   fi
else
   if [ ! -d $BUILD_DIR/build/pgmath ] ; then
      echo "ERROR: The build directory $BUILD_DIR/build/pgmath does not exist"
      echo "       run $0 without nocmake or install options. "
      exit 1
   fi
   if [ "$AOMP_BUILD_SANITIZER" == 1 ] && [ ! -d $BUILD_DIR/build/pgmath/asan ] ; then
      echo "ERROR: The build directory $BUILD_DIR/build/pgmath/asan does not exist"
      echo "       run $0 without nocmake or install options. "
      exit 1
   fi
fi

#  Need llvm-config to come from previous LLVM build
export PATH=$LLVM_INSTALL_LOC/bin:$PATH

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   if [ "$SANITIZER" != 1 ]; then
      echo
      cd $BUILD_DIR/build/pgmath
      echo " -----Running cmake ---- "
      echo ${AOMP_CMAKE} $MYCMAKEOPTS  \
           -DCMAKE_C_FLAGS="$CFLAGS -I$COMP_INC_DIR" \
           -DCMAKE_CXX_FLAGS="$CXXFLAGS -I$COMP_INC_DIR" \
           $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/runtime/libpgmath
      ${AOMP_CMAKE} $MYCMAKEOPTS  \
      -DCMAKE_C_FLAGS="$CFLAGS -I$COMP_INC_DIR" \
      -DCMAKE_CXX_FLAGS="$CXXFLAGS -I$COMP_INC_DIR" \
      $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/runtime/libpgmath  2>&1
      if [ $? != 0 ] ; then
         echo "ERROR cmake failed. Cmake flags"
         echo "      $MYCMAKEOPTS"
         exit 1
      fi
   fi

   if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
      echo
      cd $BUILD_DIR/build/pgmath/asan
      echo " -----Running cmake pgmath-asan ---- "
      echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS \
      -DCMAKE_C_FLAGS="$CFLAGS $ASAN_FLAGS" \
      -DCMAKE_CXX_FLAGS="$CXXFLAGS $ASAN_FLAGS" \
      $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/runtime/libpgmath
      ${AOMP_CMAKE} $ASAN_CMAKE_OPTS  \
      -DCMAKE_C_FLAGS="$CFLAGS $ASAN_FLAGS" \
      -DCMAKE_CXX_FLAGS="$CXXFLAGS $ASAN_FLAGS" \
      $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/runtime/libpgmath  2>&1
      if [ $? != 0 ] ; then
         echo "ERROR pgmath-asan cmake failed. Cmake flags"
         echo "      $ASAN_CMAKE_OPTS"
         exit 1
      fi
   fi
fi

if [ "$SANITIZER" != 1 ]; then
   echo
   cd $BUILD_DIR/build/pgmath
   echo " -----Running make ---- "
   echo make -j $AOMP_JOB_THREADS
   make -j $AOMP_JOB_THREADS
   if [ $? != 0 ] ; then
      echo "ERROR make -j $AOMP_JOB_THREADS failed"
      exit 1
   fi
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
echo
cd $BUILD_DIR/build/pgmath/asan
echo " -----Running make ---- "
echo make -j $AOMP_JOB_THREADS
make -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then
   echo "ERROR make -j $AOMP_JOB_THREADS failed"
   exit 1
fi
fi

if [ "$1" == "install" ] ; then
   if [ "$SANITIZER" != 1 ]; then
      cd $BUILD_DIR/build/pgmath
      echo " -----Installing to $INSTALL_FLANG ---- "
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
      echo "SUCCESSFUL INSTALL to $INSTALL_FLANG "
      echo
   fi
   if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
      cd $BUILD_DIR/build/pgmath/asan
      echo " -----Installing to $INSTALL_FLANG/lib/asan ---- "
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
      echo "SUCCESSFUL INSTALL to $INSTALL_FLANG/lib/asan"
      echo
   fi
else
   echo
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo
fi
