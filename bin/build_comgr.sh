#!/bin/bash
#
#  build_comgr.sh:  Script to build the code object manager for aomp
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_COMGR=${INSTALL_COMGR:-$AOMP_INSTALL_DIR}

REPO_DIR=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_COMGR_REPO_NAME

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo " This script builds the code object manager"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_COMGR_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/comgr"
  echo " It installs in:           $INSTALL_COMGR"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_comgr.sh                   cmake, make , NO Install "
  echo "  ./build_comgr.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_comgr.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_COMGR_REPO_NAME ] ; then
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_COMGR_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_COMGR_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_COMGR
   $SUDO touch $INSTALL_COMGR/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_COMGR"
      exit 1
   fi
   $SUDO rm $INSTALL_COMGR/testfile
fi

osversion=$(cat /etc/os-release)
#if [ "$AOMP_MAJOR_VERSION" != "12" ] && [[ "$osversion" =~ "Ubuntu 16" ]];  then
  patchrepo $REPO_DIR
#fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
  LDFLAGS="-fuse-ld=lld $ASAN_FLAGS"
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then

   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_comgr"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   BUILDTYPE="Release"
   echo $SUDO rm -rf $BUILD_AOMP/build/comgr
   $SUDO rm -rf $BUILD_AOMP/build/comgr
   export LLVM_DIR=$AOMP_INSTALL_DIR
   export Clang_DIR=$AOMP_INSTALL_DIR

   mkdir -p $BUILD_AOMP/build/comgr
   cd $BUILD_AOMP/build/comgr
   echo " -----Running comgr cmake ---- " 

   DEVICELIBS_BUILD_PATH=$AOMP_REPOS/build/AOMP_LIBDEVICE_REPO_NAME
   PACKAGE_ROOT=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_COMGR_REPO_NAME
   COMMON_PREFIX_PATH="$AOMP/include/amd_comgr;$DEVICELIBS_BUILD_PATH;$PACKAGE_ROOT;$LLVM_INSTALL_LOC"
   MYCMAKEOPTS="
      -DCMAKE_INSTALL_PREFIX='$INSTALL_COMGR'
      -DCMAKE_BUILD_TYPE=$BUILDTYPE
      -DBUILD_TESTING=OFF
      -DROCM_DIR=$AOMP_INSTALL_DIR
      -DLLVM_DIR=$AOMP_INSTALL_DIR
      -DClang_DIR=$AOMP_INSTALL_DIR"
   echo ${AOMP_CMAKE} ${MYCMAKEOPTS} -DCMAKE_PREFIX_PATH="$AOMP/lib/cmake;$COMMON_PREFIX_PATH" -DCMAKE_INSTALL_LIBDIR=lib $AOMP_ORIGIN_RPATH $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_COMGR_REPO_NAME
   ${AOMP_CMAKE} ${MYCMAKEOPTS} -DCMAKE_PREFIX_PATH="$AOMP/lib/cmake;$COMMON_PREFIX_PATH" -DCMAKE_INSTALL_LIBDIR=lib $AOMP_ORIGIN_RPATH $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_COMGR_REPO_NAME
   if [ $? != 0 ] ; then 
      echo "ERROR comgr cmake failed. cmake flags"
      exit 1
   fi

   if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
      mkdir -p $BUILD_AOMP/build/comgr/asan
      cd $BUILD_AOMP/build/comgr/asan
      echo " -----Running comgr-asan cmake ----- "
      ASAN_CMAKE_OPTS="$MYCMAKEOPTS -DCMAKE_C_COMPILER=$LLVM_INSTALL_LOC/bin/clang -DCMAKE_CXX_COMPILER=$LLVM_INSTALL_LOC/bin/clang++"
      echo ${AOMP_CMAKE} ${ASAN_CMAKE_OPTS} -DCMAKE_PREFIX_PATH="$AOMP/lib/asan/cmake;$COMMON_PREFIX_PATH:$AOMP/lib/cmake" -DCMAKE_INSTALL_LIBDIR=lib/asan $AOMP_ASAN_ORIGIN_RPATH -DCMAKE_C_FLAGS="'$ASAN_FLAGS'" -DCMAKE_CXX_FLAGS="'$ASAN_FLAGS'" $AOMP_ASAN_ORIGIN_RPATH $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_COMGR_REPO_NAME
      ${AOMP_CMAKE} ${ASAN_CMAKE_OPTS} -DCMAKE_PREFIX_PATH="$AOMP/lib/asan/cmake;$COMMON_PREFIX_PATH;$AOMP/lib/cmake" -DCMAKE_INSTALL_LIBDIR=lib/asan $AOMP_ASAN_ORIGIN_RPATH -DCMAKE_C_FLAGS="'$ASAN_FLAGS'" -DCMAKE_CXX_FLAGS="'$ASAN_FLAGS'" $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_COMGR_REPO_NAME
      if [ $? != 0 ] ; then
         echo "ERROR comgr-asan cmake failed. cmake flags"
         exit 1
      fi
   fi
fi

cd $BUILD_AOMP/build/comgr
echo
echo " -----Running make for comgr ---- " 
make -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/comgr"
      echo "  make"
      exit 1
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
   cd $BUILD_AOMP/build/comgr/asan
   echo " -----Running make for comgr-asan ---- "
   make -j $AOMP_JOB_THREADS
   if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS FAILED"
      echo "To restart:"
      echo "  cd $BUILD_AOMP/build/comgr/asan"
      echo "  make"
      exit 1
   fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/comgr
      echo " -----Installing to $INSTALL_COMGR/lib ----- " 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi

      if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
         cd $BUILD_AOMP/build/comgr/asan
         echo " -----Installing to $INSTALL_COMGR/lib/asan ----"
         $SUDO make install
         if [ $? != 0 ] ; then
            echo "ERROR make install failed"
            exit 1
         fi
      fi
      # amd_comgr.h is now in amd_comgr/amd_comgr.h, so remove depracated file
      [ -f $INSTALL_COMGR/include/amd_comgr.h ] && rm $INSTALL_COMGR/include/amd_comgr.h
      if [ "$AOMP_MAJOR_VERSION" != "12" ] && [[ "$osversion" =~ "Ubuntu 16" ]]; then
        removepatch $REPO_DIR
      fi
fi
