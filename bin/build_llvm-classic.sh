#!/bin/bash
#
#  build_llvm-classic.sh:  Script to build the classic LLVM used by flang-classic. binary driver
#         This driver will never call flang -fc1, it only calls binaries
#             clang, flang1, flang2, built elsewhere
#  Instead of downloading the ROCm 5.5 llvm package we have to
#  compile the 11vm/clang libs from source to support various
#  operating systems and spack. This will be the llvm-classic build step.
#  These libs/headers are not installed and will picked up from the build
#  tree for flang-classic.
#
BUILD_TYPE=${BUILD_TYPE:-Release}

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

if [ $AOMP_BUILD_FLANG_CLASSIC == 0 ] ; then
   if [ "$1" != "install" ] ; then
      echo "WARNING:  ROCM install for $AOMP_FLANG_CLASSIC_REL/llvm-classic not found."
      echo "          This build will skip build of flang-classic."
      echo "          The flang will link to the clang driver."
   fi
   exit
fi
TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}X86"

if [ $AOMP_STANDALONE_BUILD == 1 ] ; then
   standalone_word="_STANDALONE"
else
   standalone_word=""
fi

if [ "$AOMP_USE_NINJA" == 0 ] ; then
    AOMP_SET_NINJA_GEN=""
else
    AOMP_SET_NINJA_GEN="-G Ninja"
fi
osversion=$(cat /etc/os-release | grep -e ^VERSION_ID)
if [[ $osversion =~ '"7.' ]] || [[ $osversion =~ '"8' ]]; then
  _cxx_flag="-DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0'"
else
  _cxx_flag=""
fi

# Legacy Flang dosen't support building of compiler-rt so it
# utilizes the clang runtime libraries build/install using build_project.sh.
# The LLVM_VERSION_MAJOR of classic flang driver has to match with the clang
# binaries generated from build_project.sh.
LLVM_VERSION_MAJOR=$(${LLVM_INSTALL_LOC}/bin/clang --version | grep -oP '(?<=clang version )[0-9]+')

# We need a version of ROCM llvm that supports flang-classic 
# via the link from flang to clang.  rocm 5.5 would be best.
# This will enable removal of flang-classic driver support
# from clang to make way for flang-new.

# Options for llvm-classic cmake.
TARGETS_TO_BUILD="AMDGPU;X86"

# Do not change the AOMP_LFL_DIR default because it is the subdirectory
# from where we build the flang-classic driver binary.  This is the
# Last Frozen LLVM (LFL) for which there is amd-only clang driver support
# for flang.  Originally there was no subdirectory for LFL so setting
# AOMP_LFL_DIR to "/" would build flang-classic with the original
# ROCm 5.6 sources.
AOMP_LFL_DIR=${AOMP_LFL_DIR:-"17.0-4"}
# comment out above line and uncomment next line for new LFL
#AOMP_LFL_DIR=${AOMP_LFL_DIR:-17.0-4}

LLVMCMAKEOPTS="\
-DLLVM_ENABLE_PROJECTS=clang \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD \
-DCLANG_DEFAULT_LINKER=lld \
-DLLVM_VERSION_MAJOR="$LLVM_VERSION_MAJOR" \
-DLLVM_INCLUDE_BENCHMARKS=0 \
-DLLVM_INCLUDE_RUNTIMES=0 \
-DLLVM_INCLUDE_EXAMPLES=0 \
-DLLVM_INCLUDE_TESTS=0 \
-DLLVM_INCLUDE_DOCS=0 \
-DLLVM_INCLUDE_UTILS=0 \
-DCLANG_DEFAULT_PIE_ON_LINUX=0 \
-DLLVM_ENABLE_ZSTD=OFF \
$_cxx_flag \
$AOMP_SET_NINJA_GEN"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  help_build_aomp
fi

if [ $AOMP_STANDALONE_BUILD == 1 ] ; then
   _aomp_libllvm_stripped=${AOMP%*\/lib\/llvm}
   if [ ! -L $_aomp_libllvm_stripped ] ; then
     if [ -d $_aomp_libllvm_stripped ] ; then
        echo "ERROR: Directory $_aomp_libllvm_stripped is a physical directory."
        echo "       It must be a symbolic link or not exist"
        echo "       Specify a different value for AOMP env variable"
        exit 1
     fi
   fi
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then
   $SUDO mkdir -p $AOMP_INSTALL_DIR
   $SUDO touch $AOMP_INSTALL_DIR/testfile
   if [ $? != 0 ] ; then
      echo "ERROR: No update access to $AOMP_INSTALL_DIR"
      exit 1
   fi
   $SUDO rm $AOMP_INSTALL_DIR/testfile
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/flang-classic/$AOMP_LFL_DIR"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_DIR/build/flang-classic/$AOMP_LFL_DIR/llvm-classic
   mkdir -p $BUILD_DIR/build/flang-classic/$AOMP_LFL_DIR
   mkdir -p $BUILD_DIR/build/flang-classic/$AOMP_LFL_DIR/llvm-classic
else
   if [ ! -d $BUILD_DIR/build/flang-classic/$AOMP_LFL_DIR ] ; then
      echo "ERROR: The build directory $BUILD_DIR/build/flang-classic/$AOMP_LFL_DIR does not exist"
      echo "       run $0 without nocmake or install options. "
      exit 1
   fi
fi

# Cmake for llvm classic (ROCm 5.5).
if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   cd $BUILD_DIR/build/flang-classic/$AOMP_LFL_DIR/llvm-classic
   echo " -----Running cmake ---- "
   echo ${AOMP_CMAKE} $LLVMCMAKEOPTS  $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/flang-classic/$AOMP_LFL_DIR/llvm-classic/llvm
   ${AOMP_CMAKE} $LLVMCMAKEOPTS  $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/flang-classic/$AOMP_LFL_DIR/llvm-classic/llvm 2>&1
   if [ $? != 0 ] ; then
      echo "ERROR cmake failed. Cmake flags"
      echo "      $LLVMCMAKEOPTS"
      exit 1
   fi
   echo
fi

if [ "$1" = "cmake" ]; then
   exit 0
fi

# Build llvm classic.
echo " ---  Running $AOMP_NINJA_BIN for $BUILD_DIR/build/flang-classic/$AOMP_LFL_DIR/llvm-classic ---- "
cd $BUILD_DIR/build/flang-classic/$AOMP_LFL_DIR/llvm-classic
$AOMP_NINJA_BIN -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/flang-classic/$AOMP_LFL_DIR/llvm-classic"
      echo "  $AOMP_NINJA_BIN"
      exit 1
fi
