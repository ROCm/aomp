#!/bin/bash
# 
#  build_flang-legacy.sh:  Script to build the legacy flang binary driver
#         This driver will never call flang -fc1, it only calls binaries 
#             clang, flang1, flang2, build elsewhere
#  Instead of downloading the ROCm 5.5 llvm package we have to
#  compile the 11vm/clang libs from source to support various
#  operating systems and spack. This will be the llvm-legacy build step.
#  These libs/headers are not installed and will picked up from the build
#  tree for flang-legacy.
#
BUILD_TYPE=${BUILD_TYPE:-Release}

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

if [ $AOMP_BUILD_FLANG_LEGACY == 0 ] ; then
   if [ "$1" != "install" ] ; then
      echo "WARNING:  ROCM install for $AOMP_FLANG_LEGACY_REL/llvm-legacy not found."
      echo "          This build will skip build of flang-legacy."
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
# The LLVM_VERSION_MAJOR of legacy flang driver has to match with the clang
# binaries generated from build_project.sh.
LLVM_VERSION_MAJOR=$(${LLVM_INSTALL_LOC}/bin/clang --version | grep -oP '(?<=clang version )[0-9]+')

# We need a version of ROCM llvm that supports legacy flang 
# via the link from flang to clang.  rocm 5.5 would be best. 
# This will enable removal of legacy flang driver support 
# from clang to make way for flang-new.  

# Options for llvm-legacy cmake.
TARGETS_TO_BUILD="AMDGPU;X86"

# Do not change the AOMP_LFL_DIR default because it is the subdirectory
# from where we build the flang-legacy driver binary.  This is the
# Last Frozen LLVM (LFL) for which there is amd-only clang driver support
# for flang.  Originally there was no subdirectory for LFL so setting
# AOMP_LFL_DIR to "/" would build flang-legacy with the original
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

MYCMAKEOPTS="\
-DCMAKE_BUILD_TYPE=$BUILD_TYPE \
-DCMAKE_C_COMPILER=$LLVM_INSTALL_LOC/bin/clang \
-DCMAKE_CXX_COMPILER=$LLVM_INSTALL_LOC/bin/clang++ \
$_cxx_flag \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_LOC \
$AOMP_SET_NINJA_GEN \
"

if [ $AOMP_STANDALONE_BUILD == 1 ] ; then
  MYCMAKEOPTS="$MYCMAKEOPTS -DBUILD_SHARED_LIBS=ON $AOMP_ORIGIN_RPATH"
else
  MYCMAKEOPTS="$MYCMAKEOPTS -DBUILD_SHARED_LIBS=OFF $OPENMP_EXTRAS_ORIGIN_RPATH"
fi


if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  help_build_aomp
fi

if [ $AOMP_STANDALONE_BUILD == 1 ] ; then 
   if [ ! -L $AOMP ] ; then 
     if [ -d $AOMP ] ; then 
        echo "ERROR: Directory $AOMP is a physical directory."
        echo "       It must be a symbolic link or not exist"
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
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR
   mkdir -p $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR
   mkdir -p $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR/llvm-legacy
else
   if [ ! -d $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR ] ; then
      echo "ERROR: The build directory $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR does not exist"
      echo "       run $0 without nocmake or install options. "
      exit 1
   fi
fi

# Cmake for llvm legacy (ROCm 5.5).
if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   cd $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR/llvm-legacy
   echo " -----Running cmake ---- "
   echo ${AOMP_CMAKE} $LLVMCMAKEOPTS  $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/flang-legacy/$AOMP_LFL_DIR/llvm-legacy/llvm
   ${AOMP_CMAKE} $LLVMCMAKEOPTS  $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/flang-legacy/$AOMP_LFL_DIR/llvm-legacy/llvm 2>&1
   if [ $? != 0 ] ; then
      echo "ERROR cmake failed. Cmake flags"
      echo "      $LLVMCMAKEOPTS"
      exit 1
   fi
   echo
fi

# Build llvm legacy.
echo " ---  Running $AOMP_NINJA_BIN for $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR/llvm-legacy ---- "
cd $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR/llvm-legacy
$AOMP_NINJA_BIN -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR/llvm-legacy"
      echo "  $AOMP_NINJA_BIN"
      exit 1
fi

echo
# Cmake flang-legacy.
if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   cd $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR
   echo " -----Running cmake ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/flang-legacy/$AOMP_LFL_DIR
   ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/flang-legacy/$AOMP_LFL_DIR 2>&1
   if [ $? != 0 ] ; then 
      echo "ERROR cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi
fi

echo

# Build flang-legacy.
echo " ---  Running $AOMP_NINJA_BIN for $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR ---- "
cd $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR
$AOMP_NINJA_BIN -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/flang-legacy/$AOMP_LFL_DIR"
      echo "  $AOMP_NINJA_BIN"
      exit 1
fi

if [ "$1" == "install" ] ; then
   echo " -----Installing to $AOMP_INSTALL_DIR ---- "
   $SUDO ${AOMP_CMAKE} --build . -j $AOMP_JOB_THREADS --target install
   if [ $? != 0 ] ; then
      echo "ERROR make install failed "
      exit 1
   fi
   if [ $AOMP_BUILD_FLANG_LEGACY == 1 ] ; then
      echo "------ Linking flang-legacy to flang -------"
      if [ -L $AOMP_INSTALL_DIR/bin/flang ] ; then
         $SUDO rm $AOMP_INSTALL_DIR/bin/flang
      fi
      cd $LLVM_INSTALL_LOC/bin
      $SUDO ln -sf flang-legacy flang
   fi
   echo
   echo "SUCCESSFUL INSTALL to $AOMP_INSTALL_DIR"
   echo
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo 
fi
