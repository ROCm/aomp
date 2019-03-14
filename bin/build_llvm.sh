#!/bin/bash
# 
#  build_llvm.sh:  Script to build the llvm, clang , and lld components of the AOMP compiler. 
#                  This clang 9.0 compiler supports clang hip, OpenMP, and clang cuda
#                  offloading languages for BOTH nvidia and Radeon accelerator cards.
#                  This compiler has both the NVPTX and AMDGPU LLVM backends.
#                  The AMDGPU LLVM backend is referred to as the Lightning Compiler.
#
# See the help text below, run 'build_llvm.sh -h' for more information. 
#
BUILD_TYPE=${BUILD_TYPE:-Release}

# --- Start standard header ----
function getdname(){
   local __DIRN=`dirname "$1"`
   if [ "$__DIRN" = "." ] ; then
      __DIRN=$PWD;
   else
      if [ ${__DIRN:0:1} != "/" ] ; then
         if [ ${__DIRN:0:2} == ".." ] ; then
               __DIRN=`dirname $PWD`/${__DIRN:3}
         else
            if [ ${__DIRN:0:1} = "." ] ; then
               __DIRN=$PWD/${__DIRN:2}
            else
               __DIRN=$PWD/$__DIRN
            fi
         fi
      fi
   fi
   echo $__DIRN
}
thisdir=$(getdname $0)
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_LLVM=${INSTALL_LLVM:-$AOMP_INSTALL_DIR}

WEBSITE="http\:\/\/github.com\/ROCm-Developer-Tools\/aomp"

GCC=`which gcc`
GCPLUSCPLUS=`which g++`
if [ "$AOMP_PROC" == "ppc64le" ] ; then
   COMPILERS="-DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7"
   TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}PowerPC"
else
   COMPILERS="-DCMAKE_C_COMPILER=$GCC -DCMAKE_CXX_COMPILER=$GCPLUSCPLUS"
   if [ "$AOMP_PROC" == "aarch64" ] ; then
      TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}AArch64"
   else
      TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}X86"
   fi
fi
MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_LLVM -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD $COMPILERS -DLLVM_VERSION_SUFFIX=_AOMP_Version_$AOMP_VERSION_STRING -DBUG_REPORT_URL='https://github.com/ROCm-Developer-Tools/aomp' -DCLANG_ANALYZER_ENABLE_Z3_SOLVER=0 -DLLVM_INCLUDE_BENCHMARKS=0 -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_RPATH=$AOMP_INSTALL_DIR/lib"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  help_build_aomp
fi

if [ ! -L $AOMP ] ; then 
  if [ -d $AOMP ] ; then 
     echo "ERROR: Directory $AOMP is a physical directory."
     echo "       It must be a symbolic link or not exist"
     exit 1
  fi
fi

REPO_BRANCH=$AOMP_LLVM_REPO_BRANCH
REPO_DIR=$AOMP_REPOS/$AOMP_LLVM_REPO_NAME
checkrepo
REPO_BRANCH=$AOMP_CLANG_REPO_BRANCH
REPO_DIR=$AOMP_REPOS/$AOMP_CLANG_REPO_NAME
checkrepo
REPO_BRANCH=$AOMP_LLD_REPO_BRANCH
REPO_DIR=$AOMP_REPOS/$AOMP_LLD_REPO_NAME
checkrepo

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_LLVM
   $SUDO touch $INSTALL_LLVM/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_LLVM"
      exit 1
   fi
   $SUDO rm $INSTALL_LLVM/testfile
fi

# Fix the banner to print the AOMP version string. 
cd $AOMP_REPOS/$AOMP_LLVM_REPO_NAME
LLVMID=`git log | grep -m1 commit | cut -d" " -f2`
cd $AOMP_REPOS/$AOMP_CLANG_REPO_NAME
CLANGID=`git log | grep -m1 commit | cut -d" " -f2`
cd $AOMP_REPOS/$AOMP_LLD_REPO_NAME
LLDID=`git log | grep -m1 commit | cut -d" " -f2`
SOURCEID="Source ID:$AOMP_VERSION_STRING-$AOMP_LLVMID-$CLANGID-$LLDID"
TEMPCLFILE="/tmp/clfile$$.cpp"
ORIGCLFILE="$AOMP_REPOS/$AOMP_LLVM_REPO_NAME/lib/Support/CommandLine.cpp"
BUILDCLFILE="$BUILD_DIR/$AOMP_LLVM_REPO_NAME/lib/Support/CommandLine.cpp"
sed "s/LLVM (http:\/\/llvm\.org\/):/AOMP-${AOMP_VERSION_STRING} ($WEBSITE):\\\n $SOURCEID/" $ORIGCLFILE > $TEMPCLFILE
if [ $? != 0 ] ; then 
   echo "ERROR sed command to fix CommandLine.cpp failed."
   exit 1
fi

# Calculate the number of threads to use for make
NUM_THREADS=
if [ ! -z `which "getconf"` ]; then
   NUM_THREADS=$(`which "getconf"` _NPROCESSORS_ONLN)
   if [ "$AOMP_PROC" == "ppc64le" ] ; then
      NUM_THREADS=$(( NUM_THREADS / 2))
   fi
fi

# Skip synchronization from git repos if nocmake or install are specified
if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/$AOMP_LLVM_REPO_NAME"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_DIR/build/$AOMP_LLVM_REPO_NAME
   mkdir -p $BUILD_DIR/build/$AOMP_LLVM_REPO_NAME

   if [ $COPYSOURCE ] ; then 
      #  Copy/rsync the git repos into /tmp for faster compilation
      mkdir -p $BUILD_DIR
      echo
      echo "WARNING!  BUILD_DIR($BUILD_DIR) != AOMP_REPOS($AOMP_REPOS)"
      echo "SO RSYNCING AOMP_REPOS TO: $BUILD_DIR"
      echo
      echo rsync -a --exclude ".git" --exclude "CommandLine.cpp" --delete $AOMP_REPOS/$AOMP_LLVM_REPO_NAME $BUILD_DIR 2>&1 
      rsync -a --exclude ".git" --exclude "CommandLine.cpp" --delete $AOMP_REPOS/$AOMP_LLVM_REPO_NAME $BUILD_DIR 2>&1 
      echo rsync -a --exclude ".git" --delete $AOMP_REPOS/$AOMP_CLANG_REPO_NAME $BUILD_DIR
      rsync -a --exclude ".git" --delete $AOMP_REPOS/$AOMP_CLANG_REPO_NAME $BUILD_DIR 2>&1 
      echo rsync -a --exclude ".git" --delete $AOMP_REPOS/$AOMP_LLD_REPO_NAME $BUILD_DIR
      rsync -a --exclude ".git" --delete $AOMP_REPOS/$AOMP_LLD_REPO_NAME $BUILD_DIR 2>&1

      mkdir -p $BUILD_DIR/$AOMP_LLVM_REPO_NAMEE/tools
      if [ -L $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/clang ] ; then 
        rm $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/clang
      fi
      ln -sf $BUILD_DIR/$AOMP_CLANG_REPO_NAME $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/clang
      if [ $? != 0 ] ; then 
         echo "ERROR link command for $AOMP_CLANG_REPO_NAME to clang failed."
         exit 1
      fi
      #  Remove old ld link, now using lld
      if [ -L $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/ld ] ; then
         rm $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/ld
      fi
      if [ -L $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/lld ] ; then
        rm $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/lld
      fi
      ln -sf $BUILD_DIR/$AOMP_LLD_REPO_NAME $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/lld
      if [ $? != 0 ] ; then
         echo "ERROR link command for $AOMP_LLD_REPO_NAME to lld failed."
         exit 1
      fi

   else
      cd $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools
      rm -f $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/clang
      if [ ! -L $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/clang ] ; then
         echo ln -sf $BUILD_DIR/$AOMP_CLANG_REPO_NAME clang
         ln -sf $BUILD_DIR/$AOMP_CLANG_REPO_NAME clang
      fi
      #  Remove old ld link, now using lld
      if [ -L $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/ld ] ; then
         rm $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/ld
      fi
      if [ ! -L $BUILD_DIR/$AOMP_LLVM_REPO_NAME/tools/lld ] ; then
         echo ln -sf $BUILD_DIR/$AOMP_LLD_REPO_NAME lld
         ln -sf $BUILD_DIR/$AOMP_LLD_REPO_NAME lld
      fi
   fi

else
   if [ ! -d $BUILD_DIR/build/$AOMP_LLVM_REPO_NAME ] ; then 
      echo "ERROR: The build directory $BUILD_DIR/build/$AOMP_LLVM_REPO_NAME does not exist"
      echo "       run $0 without nocmake or install options. " 
      exit 1
   fi
fi

cd $BUILD_DIR/build/$AOMP_LLVM_REPO_NAME

if [ -f $BUILDCLFILE ] ; then 
   # only copy if there has been a change to the source.  
   diff $TEMPCLFILE $BUILDCLFILE >/dev/null
   if [ $? != 0 ] ; then 
      echo "Updating $BUILDCLFILE with corrected $SOURCEID"
      cp $TEMPCLFILE $BUILDCLFILE
   else 
      echo "File $BUILDCLFILE already has correct $SOURCEID"
   fi
else
   echo "Updating $BUILDCLFILE with $SOURCEID"
   cp $TEMPCLFILE $BUILDCLFILE
fi
rm $TEMPCLFILE

cd $BUILD_DIR/build/$AOMP_LLVM_REPO_NAME

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake ---- " 
   echo cmake $MYCMAKEOPTS  $BUILD_DIR/$AOMP_LLVM_REPO_NAME
   cmake $MYCMAKEOPTS  $BUILD_DIR/$AOMP_LLVM_REPO_NAME 2>&1
   if [ $? != 0 ] ; then 
      echo "ERROR cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi
fi

echo
echo " -----Running make ---- " 
echo make -j $NUM_THREADS 
make -j $NUM_THREADS 
if [ $? != 0 ] ; then 
   echo "ERROR make -j $NUM_THREADS failed"
   exit 1
fi

if [ "$1" == "install" ] ; then
   echo " -----Installing to $INSTALL_LLVM ---- " 
   $SUDO make install 
   if [ $? != 0 ] ; then 
      echo "ERROR make install failed "
      exit 1
   fi
   $SUDO make install/local
   if [ $? != 0 ] ; then 
      echo "ERROR make install/local failed "
      exit 1
   fi
   echo " "
   echo "------ Linking $INSTALL_LLVM to $AOMP -------"
   if [ -L $AOMP ] ; then 
      $SUDO rm $AOMP   
   fi
   $SUDO ln -sf $INSTALL_LLVM $AOMP   
   # add executables forgot by make install but needed for testing
   $SUDO cp -p $BUILD_DIR/build/$AOMP_LLVM_REPO_NAME/bin/llvm-lit $AOMP/bin/llvm-lit
   $SUDO cp -p $BUILD_DIR/build/$AOMP_LLVM_REPO_NAME/bin/FileCheck $AOMP/bin/FileCheck
   echo
   echo "SUCCESSFUL INSTALL to $INSTALL_LLVM with link to $AOMP"
   echo
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo 
fi
