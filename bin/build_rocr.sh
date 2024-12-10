#!/bin/bash
#
#  build_rocr.sh:  Script to build the rocm runtime and install into the 
#                  aomp compiler installation
#                  Requires that "build_roct.sh install" be installed first
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_ROCM=${INSTALL_ROCM:-$AOMP_INSTALL_DIR}

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds the ROCM runtime libraries"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_ROCR_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/rocr"
  echo " It installs in:           $INSTALL_ROCM"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocr.sh                   cmake, make , NO Install "
  echo "  ./build_rocr.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_rocr.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/$AOMP_ROCR_REPO_NAME ] ; then 
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_ROCR_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_ROCR_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_ROCM
   $SUDO touch $INSTALL_ROCM/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_ROCM"
      exit 1
   fi
   $SUDO rm $INSTALL_ROCM/testfile
fi

patchrepo $AOMP_REPOS/$AOMP_ROCR_REPO_NAME

if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
  LDFLAGS="-fuse-ld=lld $ASAN_FLAGS"
fi

_ompd_src_dir="$LLVM_INSTALL_LOC/share/gdb/python/ompd/src"

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then 

   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_rocr"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   BUILDTYPE="Release"
   echo rm -rf $BUILD_AOMP/build/rocr
   rm -rf $BUILD_AOMP/build/rocr
   export PATH=/opt/rocm/llvm/bin:$PATH
   MYCMAKEOPTS="-DCMAKE_INSTALL_PREFIX=$INSTALL_ROCM -DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR/lib -DIMAGE_SUPPORT=OFF $AOMP_ORIGIN_RPATH -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_C_COMPILER=${AOMP_INSTALL_DIR}/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=${AOMP_INSTALL_DIR}/lib/llvm/bin/clang++ -DLLVM_DIR=$AOMP_INSTALL_DIR/lib/llvm/bin -DBUILD_SHARED_LIBS=On"
   mkdir -p $BUILD_AOMP/build/rocr
   cd $BUILD_AOMP/build/rocr
   echo
   echo " -----Running rocr cmake ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_ROCR_REPO_NAME
   ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_ROCR_REPO_NAME
   if [ $? != 0 ] ; then 
      echo "ERROR rocr cmake failed. cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi

   if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
      # unused prefix path :$ROCM_DIR/lib/asan/cmake;${AOMP_INSTALL_DIR}/lib/cmake 
      ASAN_CMAKE_OPTS="-DCMAKE_C_COMPILER=${AOMP_INSTALL_DIR}/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=${AOMP_INSTALL_DIR}/lib/llvm/bin/clang++ -DLLVM_DIR=$AOMP_INSTALL_DIR/lib/llvm/bin -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR  -DCMAKE_INSTALL_LIBDIR=lib/asan -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR/lib -DIMAGE_SUPPORT=OFF $AOMP_ASAN_ORIGIN_RPATH -DBUILD_SHARED_LIBS=On"
      mkdir -p $BUILD_AOMP/build/rocr/asan
      cd $BUILD_AOMP/build/rocr/asan
      echo
      echo " ----Running rocr-asan cmake ----- "
      echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$ASAN_FLAGS'" -DCMAKE_CXX_FLAGS="'$ASAN_FLAGS'" $AOMP_REPOS/$AOMP_ROCR_REPO_NAME
      ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$ASAN_FLAGS'" -DCMAKE_CXX_FLAGS="'$ASAN_FLAGS'" $AOMP_REPOS/$AOMP_ROCR_REPO_NAME
      if [ $? != 0 ] ; then
         echo "ERROR rocr-asan cmake failed. cmake flags"
         echo "      $ASAN_CMAKE_OPTS"
         exit 1
      fi
   fi
   if [ "$AOMP_BUILD_DEBUG" == "1" ] ; then
      echo rm -rf $BUILD_AOMP/build/rocr_debug
      [ -d $BUILD_AOMP/build/rocr_debug ] && rm -rf $BUILD_AOMP/build/rocr_debug
      ROCR_CMAKE_OPTS="-DCMAKE_C_COMPILER=$AOMP_INSTALL_DIR/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=$AOMP_INSTALL_DIR/lib/llvm/bin/clang++ -DLLVM_DIR=$AOMP_INSTALL_DIR/lib/llvm/bin -DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR/lib -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR -DCMAKE_BUILD_TYPE=Debug $AOMP_DEBUG_ORIGIN_RPATH -DCMAKE_INSTALL_LIBDIR=lib-debug -DBUILD_SHARED_LIBS=On"
      echo  
      echo " -----Running rocr_debug cmake -----"
      mkdir -p  $BUILD_AOMP/build/rocr_debug
      cd $BUILD_AOMP/build/rocr_debug
      _prefix_map="\""-fdebug-prefix-map=$AOMP_REPOS/$AOMP_ROCR_REPO_NAME/src=$_ompd_src_dir/rocr/src"\"" 
      echo ${AOMP_CMAKE} $ROCR_CMAKE_OPTS -DCMAKE_C_FLAGS="-g $_prefix_map" -DCMAKE_CXX_FLAGS="-g $_prefix_map" $AOMP_REPOS/$AOMP_ROCR_REPO_NAME
      ${AOMP_CMAKE} $ROCR_CMAKE_OPTS -DCMAKE_C_FLAGS="-g $_prefix_map" -DCMAKE_CXX_FLAGS="-g $_prefix_map" $AOMP_REPOS/$AOMP_ROCR_REPO_NAME
      if [ $? != 0 ] ; then
         echo "ERROR rocr_debug cmake failed.cmake flags"
         echo "      $ROCR_CMAKE_OPTS"
         exit 1
      fi
   fi
fi

if [ "$1" = "cmake" ]; then
   exit 0
fi

cd $BUILD_AOMP/build/rocr
echo
echo " -----Running make for rocr ---- " 
echo make -j $AOMP_JOB_THREADS
make -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/rocr"
      echo "  make"
      exit 1
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
   cd $BUILD_AOMP/build/rocr/asan
   echo
   echo " -----Running make for rocr-asan ---- "
   echo make -j $AOMP_JOB_THREADS
   make -j $AOMP_JOB_THREADS
   if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS FAILED"
      echo "To restart:"
      echo "  cd $BUILD_AOMP/build/rocr/asan"
      echo "  make"
      exit 1
   fi
fi
if [ "$AOMP_BUILD_DEBUG" == 1 ] ; then
   cd $BUILD_AOMP/build/rocr_debug
   echo
   echo " ----- Running make for rocr_debug ----- "
   make -j $AOMP_JOB_THREADS
   if [ $? != 0 ] ; then
     echo " "
     echo "ERROR: make -j $AOMP_JOB_THREADS FAILED"
     echo "To restart:"
     echo "  cd $BUILD_AOMP/build/rocr_debug"
     echo "  make"
     exit 1
   fi
fi
#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/rocr
      echo " -----Installing to $INSTALL_ROCM/lib ----- " 
      echo $SUDO make install 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi

      if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
         cd $BUILD_AOMP/build/rocr/asan
         echo " ------Installing to $INSTALL_ROCM/lib/asan ------ "
         echo $SUDO make install
         $SUDO make install
         if [ $? != 0 ] ; then
            echo "ERROR make install failed "
            exit 1
         fi
      fi
      if [ "$AOMP_BUILD_DEBUG" == 1 ] ; then
         cd $BUILD_AOMP/build/rocr_debug
         echo " -----Installing to $INSTALL_ROCM/lib-debug ----- "
         $SUDO make install
         if [ $? != 0 ] ; then
            echo "ERROR make install for rocr  failed "
            exit 1
         fi
	$SUDO mkdir -p $_ompd_src_dir/rocr
        $SUDO cp -r $AOMP_REPOS/$AOMP_ROCR_REPO_NAME/src $_ompd_src_dir/rocr
        # remove non-source files to save space
        find $_ompd_src_dir/rocr/src  -type f  | grep  -v "\.cpp$\|\.h$\|\.hpp$" | xargs rm
        rm -rf $_ompd_src_dir/rocr/src/RPM
        rm -rf $_ompd_src_dir/rocr/src/DEBIAN
        rm -rf $_ompd_src_dir/rocr/src/cmake_modules
      fi
      removepatch $AOMP_REPOS/$AOMP_ROCR_REPO_NAME
fi
