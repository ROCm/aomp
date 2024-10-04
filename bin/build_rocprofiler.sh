#!/bin/bash
#
#  build_rocprofiler.sh:  Script to build rocprofiler for AOMP standalone build
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_ROCPROF=${INSTALL_ROCPROF:-$AOMP_INSTALL_DIR}
export HIP_CLANG_PATH=$LLVM_INSTALL_LOC/bin
export ROCM_PATH=$AOMP_INSTALL_DIR

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds the ROCM runtime libraries"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_PROF_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/rocprofiler"
  echo " It installs in:           $INSTALL_ROCPROF"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocprofiler.sh                   cmake, make , NO Install "
  echo "  ./build_rocprofiler.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_rocprofiler.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/$AOMP_PROF_REPO_NAME ] ; then 
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_PROF_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_PROF_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_ROCPROF
   $SUDO touch $INSTALL_ROCPROF/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_ROCPROF"
      exit 1
   fi
   $SUDO rm $INSTALL_ROCPROF/testfile
fi

patchrepo $AOMP_REPOS/$AOMP_PROF_REPO_NAME

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then 
   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_rocprofiler"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   echo rm -rf $BUILD_AOMP/build/rocprofiler
   rm -rf $BUILD_AOMP/build/rocprofiler
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
   BUILD_TYPE="Release"
   export CMAKE_BUILD_TYPE=$BUILD_TYPE
   CMAKE_PREFIX_PATH="$ROCM_DIR/roctracer/include/ext;$ROCM_DIR/include/platform;$ROCM_DIR/include;$ROCM_DIR/lib;$ROCM_DIR;$HOME/.local/bin;$HOME/local/aqlprofile;$LLVM_INSTALL_LOC"
   export CMAKE_PREFIX_PATH
   #export HSA_RUNTIME_INC=$ROCM_DIR/include
   #export HSA_RUNTIME_LIB=$ROCM_DIR/include/lib
   #export HSA_KMT_LIB=$ROCM_DIR/lib 
   #export HSA_KMT_LIB_PATH=$ROCM_DIR/lib 
   GFXSEMICOLONS=`echo $GFXLIST | tr ' ' ';' `
   mkdir -p $BUILD_AOMP/build/rocprofiler
   cd $BUILD_AOMP/build/rocprofiler
   export PATH=$HOME/.local/bin:$INSTALL_ROCPROF/bin:$PATH
   echo " -----Running rocprofiler cmake ---- " 
   echo ${AOMP_CMAKE} -DCMAKE_INSTALL_LIBDIR=lib -DENABLE_ASAN_PACKAGING=ON -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DROCM_PATH=$AOMP_INSTALL_DIR -DCMAKE_MODULE_PATH=$INSTALL_ROCPROF/lib/cmake/hip -DCMAKE_INSTALL_PREFIX=$INSTALL_ROCPROF -DCMAKE_PREFIX_PATH="""$CMAKE_PREFIX_PATH""" $CMAKE_WITH_EXPERIMENTAL $AOMP_ORIGIN_RPATH -DGPU_TARGETS="""$GFXSEMICOLONS""" -DPROF_API_HEADER_PATH=$INSTALL_ROCPROF/include/roctracer/ext  -DHIP_ROOT_DIR=$INSTALL_ROCPROF/hip $AOMP_REPOS/$AOMP_PROF_REPO_NAME -DAQLPROFILE_LIB=$AOMP_SUPP/aqlprofile/lib/libhsa-amd-aqlprofile64.so -DCMAKE_CXX_FLAGS=-I$HOME/local/rocmsmilib/include -DHIP_HIPCC_FLAGS="-I$HOME/local/rocmsmilib/include" -DCMAKE_EXE_LINKER_FLAGS="-L$HOME/local/rocmsmilib/lib -L$HOME/local/rocmsmilib/lib64 -Wl,--disable-new-dtags"
   ${AOMP_CMAKE} -DCMAKE_INSTALL_LIBDIR=lib -DENABLE_ASAN_PACKAGING=ON -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DROCM_PATH=$AOMP_INSTALL_DIR -DCMAKE_MODULE_PATH=$INSTALL_ROCPROF/lib/cmake/hip -DCMAKE_INSTALL_PREFIX=$INSTALL_ROCPROF -DCMAKE_PREFIX_PATH="""$CMAKE_PREFIX_PATH""" $CMAKE_WITH_EXPERIMENTAL $AOMP_ORIGIN_RPATH -DGPU_TARGETS="""$GFXSEMICOLONS""" -DPROF_API_HEADER_PATH=$INSTALL_ROCPROF/include/roctracer/ext  -DHIP_ROOT_DIR=$INSTALL_ROCPROF/hip $AOMP_REPOS/$AOMP_PROF_REPO_NAME -DAQLPROFILE_LIB=$AOMP_SUPP/aqlprofile/lib/libhsa-amd-aqlprofile64.so -DCMAKE_CXX_FLAGS=-I$HOME/local/rocmsmilib/include -DHIP_HIPCC_FLAGS="-I$HOME/local/rocmsmilib/include" -DCMAKE_EXE_LINKER_FLAGS="-L$HOME/local/rocmsmilib/lib -L$HOME/local/rocmsmilib/lib64 -Wl,--disable-new-dtags"
   if [ $? != 0 ] ; then 
      echo "ERROR rocprofiler cmake failed. cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi
fi

cd $BUILD_AOMP/build/rocprofiler
echo
echo " -----Running make for rocprofiler ---- " 
echo make -j $AOMP_JOB_THREADS
make -j $AOMP_JOB_THREADS
make -j $AOMP_JOB_THREADS mytest
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/rocprofiler"
      echo "  make"
      exit 1
fi

doxygen=`which doxygen`
if [ ! -z $doxygen ] ; then
   # the rocprofiler CMakeLists.txt will prepare docs install if doxygen found.
   # However, the make doc has issues.  But if you dont make doc, the install
   # fails.  This 'make doc' will do enough so install does not fail.
   echo make -j $AOMP_JOB_THREADS doc
   make -j $AOMP_JOB_THREADS doc 2>/dev/null >/dev/null
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/rocprofiler
      echo " -----Installing to $INSTALL_ROCPROF/lib ----- " 
      echo $SUDO make install 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
      removepatch $AOMP_REPOS/$AOMP_PROF_REPO_NAME
else
   echo
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo
fi
