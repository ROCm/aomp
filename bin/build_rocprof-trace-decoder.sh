#!/bin/bash
#
#  build_rocprof-trace-decoder.sh:  Script to build rocprof-trace-decoder for AOMP standalone build
#

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/aomp_common_vars"
# --- end standard header ----

INSTALL_ROCPROF_TRACE_DECODER=${INSTALL_ROCPROF_TRACE_DECODER:-$AOMP_INSTALL_DIR}
export HIP_CLANG_PATH=$LLVM_INSTALL_LOC/bin
export ROCM_PATH=$AOMP_INSTALL_DIR

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo " This script builds the ROCM runtime libraries"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_PROF_TRACE_DECODER_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/rocprof-trace-decoder"
  echo " It installs in:           $INSTALL_ROCPROF_TRACE_DECODER"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocprof-trace-decoder.sh                   cmake, make , NO Install "
  echo "  ./build_rocprof-trace-decoder.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_rocprof-trace-decoder.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit
fi

if [ ! -d "$AOMP_REPOS/$AOMP_PROF_TRACE_DECODER_REPO_NAME" ] ; then
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_PROF_TRACE_DECODER_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_PROF_TRACE_DECODER_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then
   $SUDO mkdir -p "$INSTALL_ROCPROF_TRACE_DECODER"
   if ! $SUDO touch "$INSTALL_ROCPROF_TRACE_DECODER"/testfile; then
      echo "ERROR: No update access to $INSTALL_ROCPROF_TRACE_DECODER"
      exit 1
   fi
   $SUDO rm "$INSTALL_ROCPROF_TRACE_DECODER"/testfile
fi

patchrepo "$AOMP_REPOS/rocprof-trace-decoder"

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo " "
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_rocprof-trace-decoder"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   echo "rm -rf $BUILD_AOMP/build/rocprof-trace-decoder"
   rm -rf "$BUILD_AOMP"/build/rocprof-trace-decoder

   BUILD_TYPE="Release"

   mkdir -p "$BUILD_AOMP"/build/rocprof-trace-decoder
   cd "$BUILD_AOMP"/build/rocprof-trace-decoder || exit
   export PATH=$HOME/.local/bin:$INSTALL_ROCPROF_TRACE_DECODER/bin:$PATH

   declare -a MYCMAKEOPTS

   MYCMAKEOPTS=(-DLLVM_DIR="$AOMP_INSTALL_DIR/lib/llvm/lib/cmake/llvm"
	        -DCMAKE_INSTALL_PREFIX="$INSTALL_ROCPROF_TRACE_DECODER"
	        -DCMAKE_BUILD_TYPE="$BUILD_TYPE")

   echo " -----Running rocprof-trace-decoder cmake ---- "
   echo "${AOMP_CMAKE}" "${MYCMAKEOPTS[@]}" "$AOMP_REPOS/$AOMP_PROF_TRACE_DECODER_REPO_NAME"
   if ! ${AOMP_CMAKE} "${MYCMAKEOPTS[@]}" "$AOMP_REPOS/$AOMP_PROF_TRACE_DECODER_REPO_NAME"; then
      echo "ERROR rocprof-trace-decoder cmake failed. cmake flags"
      echo "      $(shquot "${MYCMAKEOPTS[@]}")"
      exit 1
   fi
fi

if [ "$1" = "cmake" ]; then
   exit 0
fi

cd "$BUILD_AOMP"/build/rocprof-trace-decoder || exit
echo
echo " -----Running make for rocprof-trace-decoder ---- "
echo make -j "$AOMP_JOB_THREADS"
if ! make -j "$AOMP_JOB_THREADS"; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_AOMP/build/rocprof-trace-decoder"
      echo "  make"
      exit 1
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
      cd "$BUILD_AOMP"/build/rocprof-trace-decoder || exit
      echo " -----Installing to $INSTALL_ROCPROF_TRACE_DECODER/lib ----- "
      echo $SUDO make install
      if ! $SUDO make install; then
         echo "ERROR make install failed "
         exit 1
      fi

      removepatch "$AOMP_REPOS/rocprof-trace-decoder"
else
   echo
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo
fi
