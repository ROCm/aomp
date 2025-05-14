#!/bin/bash
#
#  build_rocprofiler-sdk.sh:  Script to build rocprofiler-sdk for AOMP standalone build
#

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/aomp_common_vars"
# --- end standard header ----

INSTALL_ROCPROF_SDK=${INSTALL_ROCPROF_SDK:-$AOMP_INSTALL_DIR}
export HIP_CLANG_PATH=$LLVM_INSTALL_LOC/bin
export ROCM_PATH=$AOMP_INSTALL_DIR

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo " This script builds the ROCM runtime libraries"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_PROF_SDK_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/rocprofiler-sdk"
  echo " It installs in:           $INSTALL_ROCPROF_SDK"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocprofiler-sdk.sh                   cmake, make , NO Install "
  echo "  ./build_rocprofiler-sdk.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_rocprofiler-sdk.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit
fi

if [ ! -d "$AOMP_REPOS/$AOMP_PROF_SDK_REPO_NAME" ] ; then
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_PROF_SDK_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_PROF_SDK_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then
   $SUDO mkdir -p "$INSTALL_ROCPROF_SDK"
   if ! $SUDO touch "$INSTALL_ROCPROF_SDK"/testfile; then
      echo "ERROR: No update access to $INSTALL_ROCPROF_SDK"
      exit 1
   fi
   $SUDO rm "$INSTALL_ROCPROF_SDK"/testfile
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo " "
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_rocprofiler-sdk"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   echo "rm -rf $BUILD_AOMP/build/rocprofiler-sdk"
   rm -rf "$BUILD_AOMP"/build/rocprofiler-sdk

   BUILD_TYPE="Release"
   GFXSEMICOLONS=$(echo "$GFXLIST" | tr ' ' ';')

   mkdir -p "$BUILD_AOMP"/build/rocprofiler-sdk
   cd "$BUILD_AOMP"/build/rocprofiler-sdk || exit
   export PATH=$HOME/.local/bin:$INSTALL_ROCPROF_SDK/bin:$PATH

   declare -a MYCMAKEOPTS

   MYCMAKEOPTS=(-DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR"
                -DCMAKE_INSTALL_PREFIX="$INSTALL_ROCPROF_SDK"
	        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
	        -DROCM_ROOT_DIR="$AOMP_INSTALL_DIR"
	        -DBUILD_SHARED_LIBS=On
	        -DGPU_TARGETS="$GFXSEMICOLONS"
	        -DROCPROFILER_BUILD_SAMPLES=ON
		-DROCPROFILER_BUILD_TESTS=ON)

   echo " -----Running rocprofiler-sdk cmake ---- "
   echo "${AOMP_CMAKE}" "${MYCMAKEOPTS[@]}" "$AOMP_REPOS/$AOMP_PROF_SDK_REPO_NAME"
   if ! ${AOMP_CMAKE} "${MYCMAKEOPTS[@]}" "$AOMP_REPOS/$AOMP_PROF_SDK_REPO_NAME"; then
      echo "ERROR rocprofiler-sdk cmake failed. cmake flags"
      echo "      $(shquot "${MYCMAKEOPTS[@]}")"
      exit 1
   fi
fi

if [ "$1" = "cmake" ]; then
   exit 0
fi

cd "$BUILD_AOMP"/build/rocprofiler-sdk || exit
echo
echo " -----Running make for rocprofiler-sdk ---- "
echo make -j "$AOMP_JOB_THREADS"
if ! make -j "$AOMP_JOB_THREADS"; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_AOMP/build/rocprofiler-sdk"
      echo "  make"
      exit 1
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
      cd "$BUILD_AOMP"/build/rocprofiler-sdk || exit
      echo " -----Installing to $INSTALL_ROCPROF_SDK/lib ----- "
      echo $SUDO make install
      if ! $SUDO make install; then
         echo "ERROR make install failed "
         exit 1
      fi
else
   echo
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo
fi
