#!/bin/bash
#
#  build_roct.sh:  Script to build the ROCt thunk libraries.
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_ROCT=${INSTALL_ROCT:-$AOMP_INSTALL_DIR}

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds the ROCt thunk libraries"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_ROCT_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/roct"
  echo " It installs in:           $INSTALL_ROCT"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_roct.sh                   cmake, make , NO Install "
  echo "  ./build_roct.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_roct.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/$AOMP_ROCT_REPO_NAME ] ; then 
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_ROCT_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_ROCT_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_ROCT
   $SUDO touch $INSTALL_ROCT/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_ROCT"
      exit 1
   fi
   $SUDO rm $INSTALL_ROCT/testfile
fi

patchrepo $AOMP_REPOS/$AOMP_ROCT_REPO_NAME

if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
   ASAN_LIB_PATH=$($AOMP_CLANG_COMPILER --print-runtime-dir)
   ASAN_FLAGS="-g -fsanitize=address -shared-libasan -Wl,-rpath=$ASAN_LIB_PATH -L$ASAN_LIB_PATH"
   LDFLAGS="-fuse-ld=lld $ASAN_FLAGS"
fi

_install_src_dir_roct="$AOMP_INSTALL_DIR/share/gdb/python/ompd/src/roct"
if [ "$AOMP_BUILD_DEBUG" == 1 ] ; then
   ROCT_DEBUG_FLAGS="-g -DOPENMP_SOURCE_DEBUG_MAP=-fdebug-prefix-map=$AOMP_REPOS/$AOMP_ROCT_REPO_NAME/src=$_install_src_dir_roct/src"
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then 

   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_roct"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   BUILDTYPE="Release"
   echo $SUDO rm -rf $BUILD_AOMP/build/roct
   $SUDO rm -rf $BUILD_AOMP/build/roct
   MYCMAKEOPTS="-DCMAKE_INSTALL_PREFIX=$INSTALL_ROCT -DCMAKE_BUILD_TYPE=$BUILDTYPE $AOMP_ORIGIN_RPATH -DCMAKE_INSTALL_LIBDIR=$AOMP_INSTALL_DIR/lib"
   mkdir -p $BUILD_AOMP/build/roct
   cd $BUILD_AOMP/build/roct
   echo " -----Running roct cmake ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_ROCT_REPO_NAME
   ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_ROCT_REPO_NAME
   if [ $? != 0 ] ; then 
      echo "ERROR roct cmake failed. cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi

   if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
      mkdir -p $BUILD_AOMP/build/roct/asan
      cd $BUILD_AOMP/build/roct/asan
      ASAN_CMAKE_OPTS="-DCMAKE_C_COMPILER=$AOMP_CLANG_COMPILER -DCMAKE_CXX_COMPILER=$AOMP_CLANGXX_COMPILER -DCMAKE_INSTALL_PREFIX=$INSTALL_ROCT -DCMAKE_BUILD_TYPE=$BUILDTYPE $AOMP_ORIGIN_RPATH -DCMAKE_INSTALL_LIBDIR=$AOMP_INSTALL_DIR/lib/asan"
      echo " -----Running roct-asan cmake -----"
      echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$ASAN_FLAGS'" -DCMAKE_CXX_FLAGS="'$ASAN_FLAGS'" $AOMP_REPOS/$AOMP_ROCT_REPO_NAME
      ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$ASAN_FLAGS'" -DCMAKE_CXX_FLAGS="'$ASAN_FLAGS'" $AOMP_REPOS/$AOMP_ROCT_REPO_NAME
      if [ $? != 0 ] ; then
         echo "ERROR roct-asan cmake failed.cmake flags"
         echo "      $ASAN_CMAKE_OPTS"
         exit 1
      fi
   fi
   if [ "$AOMP_BUILD_DEBUG" == "1" ] ; then
      echo rm -rf $BUILD_DIR/build/roct_debug
      [ -d $BUILD_DIR/build/roct_debug ] && rm -rf $BUILD_DIR/build/roct_debug
      ROCT_CMAKE_OPTS="-DCMAKE_C_COMPILER=$AOMP_CLANG_COMPILER -DCMAKE_CXX_COMPILER=$AOMP_CLANGXX_COMPILER -DCMAKE_INSTALL_PREFIX=$INSTALL_ROCT -DCMAKE_BUILD_TYPE=$BUILDTYPE $AOMP_ORIGIN_RPATH -DCMAKE_INSTALL_LIBDIR=$AOMP_INSTALL_DIR/lib-debug"
      echo " -----Running roct_debug cmake -----"
      mkdir -p  $BUILD_AOMP/build/roct_debug
      cd $BUILD_AOMP/build/roct_debug
      echo ${AOMP_CMAKE} $ROCT_CMAKE_OPTS -DCMAKE_C_FLAGS="'$ROCT_DEBUG_FLAGS'" -DCMAKE_CXX_FLAGS="'$ROCT_DEBUG_FLAGS'" $AOMP_REPOS/$AOMP_ROCT_REPO_NAME
      ${AOMP_CMAKE} $ROCT_CMAKE_OPTS -DCMAKE_C_FLAGS="'$ROCT_DEBUG_FLAGS'" -DCMAKE_CXX_FLAGS="'$ROCT_DEBUG_FLAGS'" $AOMP_REPOS/$AOMP_ROCT_REPO_NAME
      if [ $? != 0 ] ; then
         echo "ERROR roct_debug cmake failed.cmake flags"
         echo "      $ROCT_CMAKE_OPTS"
         exit 1
      fi
   fi
fi

cd $BUILD_AOMP/build/roct
echo
echo " -----Running make for roct ---- " 
make -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/roct"
      echo "  make"
      exit 1
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
   echo
   echo " ----- Running make for roct-asan ----- "
   make -j $AOMP_JOB_THREADS
   if [ $? != 0 ] ; then
     echo " "
     echo "ERROR: make -j $AOMP_JOB_THREADS FAILED"
     echo "To restart:"
     echo "  cd $BUILD_AOMP/build/roct/asan "
     echo "  make"
     exit 1
   fi
fi

if [ "$AOMP_BUILD_DEBUG" == 1 ] ; then
   cd $BUILD_AOMP/build/roct_debug
   echo
   echo " ----- Running make for roct_debug ----- "
   make -j $AOMP_JOB_THREADS
   if [ $? != 0 ] ; then
     echo " "
     echo "ERROR: make -j $AOMP_JOB_THREADS FAILED"
     echo "To restart:"
     echo "  cd $BUILD_AOMP/build/roct_debug"
     echo "  make"
     exit 1
   fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/roct
      echo " -----Installing to $INSTALL_ROCT/lib ----- " 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi

      if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
         cd $BUILD_AOMP/build/roct/asan
         echo " -----Installing to $INSTALL_ROCT/lib/asan ----- "
         $SUDO make install
         if [ $? != 0 ] ; then
            echo "ERROR make install failed "
            exit 1
         fi
      fi

      if [ "$AOMP_BUILD_DEBUG" == 1 ] ; then
         cd $BUILD_AOMP/build/roct_debug
         echo " -----Installing to $INSTALL_ROCT/lib-debug ----- "
         $SUDO make install
         if [ $? != 0 ] ; then
            echo "ERROR make install for roct  failed "
            exit 1
         fi
	 $SUDO mkdir -p $_install_src_dir_roct
         echo $SUDO cp -r $AOMP_REPOS/$AOMP_ROCT_REPO_NAME/src $_install_src_dir_roct
         $SUDO cp -r $AOMP_REPOS/$AOMP_ROCT_REPO_NAME/src $_install_src_dir_roct
      fi

      removepatch $AOMP_REPOS/$AOMP_ROCT_REPO_NAME
fi
