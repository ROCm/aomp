#!/bin/bash
#
#  build_rocprofiler-register.sh:  Script to build rocprofiler-register for AOMP standalone build
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_ROCPROF_REGISTER=${INSTALL_ROCPROF_REGISTER:-$AOMP_INSTALL_DIR}
export HIP_CLANG_PATH=$INSTALL_ROCPROF_REGISTER/bin

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds the ROCM runtime libraries"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_PROF_REGISTER_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/$AOMP_PROF_REGISTER_REPO_NAME"
  echo " It installs in:           $INSTALL_ROCPROF_REGISTER"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocprofiler-register.sh                   cmake, make , NO Install "
  echo "  ./build_rocprofiler-register.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_rocprofiler-register.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/$AOMP_PROF_REGISTER_REPO_NAME ] ; then 
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_PROF_REGISTER_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_PROF_REGISTER_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_ROCPROF_REGISTER
   $SUDO touch $INSTALL_ROCPROF_REGISTER/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_ROCPROF_REGISTER"
      exit 1
   fi
   $SUDO rm $INSTALL_ROCPROF_REGISTER/testfile
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then 
   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/$AOMP_PROF_REGISTER_REPO_NAME"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   echo rm -rf $BUILD_AOMP/build/$AOMP_PROF_REGISTER_REPO_NAME
   rm -rf $BUILD_AOMP/build/$AOMP_PROF_REGISTER_REPO_NAME
   BUILD_TYPE="Release"
   export CMAKE_BUILD_TYPE=$BUILD_TYPE
   CMAKE_PREFIX_PATH="$ROCM_DIR/include;$ROCM_DIR/lib;$ROCM_DIR"
   export CMAKE_PREFIX_PATH
   GFXSEMICOLONS=`echo $GFXLIST | tr ' ' ';' `
   mkdir -p $BUILD_AOMP/build/$AOMP_PROF_REGISTER_REPO_NAME
   cd $BUILD_AOMP/build/$AOMP_PROF_REGISTER_REPO_NAME
   echo " -----Running $AOMP_PROF_REGISTER_REPO_NAME cmake ---- " 
   echo ${AOMP_CMAKE} -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DROCM_PATH=$AOMP_INSTALL_DIR -DCMAKE_INSTALL_PREFIX=$INSTALL_ROCPROF_REGISTER -DCMAKE_PREFIX_PATH="""$CMAKE_PREFIX_PATH""" -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags" -DROCPROFILER_REGISTER_BUILD_TESTS=1 -DROCPROFILER_REGISTER_BUILD_SAMPLES=1 -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags" -DBUILD_SHARED_LIBS=ON -DENABLE_LDCONFIG=OFF -DROCPROFILER_REGISTER_BUILD_TESTS=1 -DROCPROFILER_REGISTER_BUILD_SAMPLES=1 $AOMP_ORIGIN_RPATH $AOMP_REPOS/$AOMP_PROF_REGISTER_REPO_NAME
   ${AOMP_CMAKE} -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DROCM_PATH=$AOMP_INSTALL_DIR -DCMAKE_INSTALL_PREFIX=$INSTALL_ROCPROF_REGISTER -DCMAKE_PREFIX_PATH="""$CMAKE_PREFIX_PATH""" $AOMP_ORIGIN_RPATH -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags" -DBUILD_SHARED_LIBS=ON -DENABLE_LDCONFIG=OFF -DROCPROFILER_REGISTER_BUILD_TESTS=1 -DROCPROFILER_REGISTER_BUILD_SAMPLES=1 $AOMP_REPOS/$AOMP_PROF_REGISTER_REPO_NAME

   if [ $? != 0 ] ; then 
      echo "ERROR $AOMP_PROF_REGISTER_REPO_NAME cmake failed. cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi
fi

if [ "$1" = "cmake" ]; then
   exit 0
fi

cd $BUILD_AOMP/build/$AOMP_PROF_REGISTER_REPO_NAME
echo
echo " -----Running make for $AOMP_PROF_REGISTER_REPO_NAME ---- " 
echo make -j $AOMP_JOB_THREADS
make -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/$AOMP_PROF_REGISTER_REPO_NAME"
      echo "  make"
      exit 1
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/$AOMP_PROF_REGISTER_REPO_NAME
      echo " -----Installing to $INSTALL_ROCPROF_REGISTER/lib ----- " 
      echo $SUDO make install 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
else
   echo
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo
fi
