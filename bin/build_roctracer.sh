#!/bin/bash
#
#  build_roctracer.sh:  Script to build roctracer for AOMP standalone build
#

# --- Start standard header to set AOMP environment variables ----

realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_ROCTRACE=${INSTALL_ROCTRACE:-$AOMP_INSTALL_DIR}
export HIP_CLANG_PATH=$LLVM_INSTALL_LOC/bin
echo $HIP_CLANG_PATH
export ROCM_PATH=$ROCM_DIR

# Needed for systems that have both AMD and Nvidia cards installed.
HIP_PLATFORM=${HIP_PLATFORM:-amd}
export HIP_PLATFORM

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds the ROCM runtime libraries"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_TRACE_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/roctracer"
  echo " It installs in:           $INSTALL_ROCTRACE"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_roctracer.sh                   cmake, make , NO Install "
  echo "  ./build_roctracer.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_roctracer.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/$AOMP_TRACE_REPO_NAME ] ; then 
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_TRACE_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_TRACE_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_ROCTRACE
   $SUDO touch $INSTALL_ROCTRACE/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_ROCTRACE"
      exit 1
   fi
   $SUDO rm $INSTALL_ROCTRACE/testfile
fi

patchrepo $AOMP_REPOS/$AOMP_TRACE_REPO_NAME

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then 
   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_roctracer"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   echo rm -rf $BUILD_AOMP/build/roctracer
   rm -rf $BUILD_AOMP/build/roctracer
   CMAKE_WITH_EXPERIMENTAL=""
   if [ -d "/usr/include/c++/5/experimental" ] ; then
      _loc=`which gcc`
      if [ "$_loc" != "" ] ; then
         _gccver=`$_loc --version | grep gcc | cut -d")" -f2 | cut -d"." -f1`
	 if [ "$_gccver" == "5" ] ; then
            CMAKE_WITH_EXPERIMENTAL="-DCMAKE_CXX_FLAGS=-I/usr/include/c++/5/experimental"
         fi
      fi
   fi
   BUILD_TYPE="release"
   export CMAKE_BUILD_TYPE=$BUILD_TYPE
   CMAKE_PREFIX_PATH="$ROCM_DIR/include/hsa;$ROCM_DIR/include;$ROCM_DIR/lib;$ROCM_DIR;$LLVM_INSTALL_LOC"
   export CMAKE_PREFIX_PATH
   GFXSEMICOLONS=`echo $GFXLIST | tr ' ' ';' `
   export ROCM_PATH=$ROCM_DIR
   export HIPCC_COMPILE_FLAGS_APPEND="--rocm-path=$ROCM_PATH"
   mkdir -p $BUILD_AOMP/build/roctracer
   cd $BUILD_AOMP/build/roctracer
   echo " -----Running roctracer cmake ---- " 
   ${AOMP_CMAKE} -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_ROCTRACE -DCMAKE_PREFIX_PATH="""$CMAKE_PREFIX_PATH""" $CMAKE_WITH_EXPERIMENTAL $AOMP_ORIGIN_RPATH -DGPU_TARGETS="""$GFXSEMICOLONS""" -DROCM_PATH=$ROCM_DIR $AOMP_REPOS/$AOMP_TRACE_REPO_NAME
   if [ $? != 0 ] ; then 
      echo "ERROR roctracer cmake failed. cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi
fi

cd $BUILD_AOMP/build/roctracer
echo
echo " -----Running make for roctracer ---- " 
echo make -j $AOMP_JOB_THREADS
make -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/roctracer"
      echo "  make"
      exit 1
fi

doxygen=`which doxygen`
if [ ! -z $doxygen ] ; then
   # the roctracer CMakeLists.txt will prepare docs install if doxygen found.
   # However, the make doc has issues.  But if you dont make doc, the install
   # fails.  This 'make doc' will do enough so install does not fail.
   echo make -j $AOMP_JOB_THREADS doc
   make -j $AOMP_JOB_THREADS doc 2>/dev/null >/dev/null
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/roctracer
      echo " -----Installing to $INSTALL_ROCTRACE/lib ----- " 
      echo $SUDO make install 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
      removepatch $AOMP_REPOS/$AOMP_TRACE_REPO_NAME
else
   echo
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo
fi
