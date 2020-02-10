#!/bin/bash
# 
#  build_project_with_externals.sh:  Script to build llvm compiler with external
#     components build by LLVM. The goal is to get a single build (cmake, make,
#     make install) for LLVM with this single build script.  This script also 
#     uses consistent CMAKE controls for the AOCC compiler. 
#
#  This clang 10.0 compiler supports clang hip, OpenMP, and clang cuda
#  offloading languages for BOTH nvidia and Radeon accelerator cards.
#  This compiler has both the NVPTX and AMDGPU LLVM backends.
#  The AMDGPU LLVM backend is referred to as the Lightning Compiler.
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


GCC_LOC=`which gcc`
GCPLUSCPLUS_LOC=`which g++`
if [ "$AOMP_PROC" == "ppc64le" ] ; then
   COMPILERS="-DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7"
   TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}PowerPC"
else
   COMPILERS="-DCMAKE_C_COMPILER=$GCC_LOC -DCMAKE_CXX_COMPILER=$GCPLUSCPLUS_LOC"
   if [ "$AOMP_PROC" == "aarch64" ] ; then
      TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}AArch64"
   else
      TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}X86"
   fi
fi

# When building from release source (no git), turn off test items that are not distributed
if [ "$AOMP_CHECK_GIT_BRANCH" == 1 ] ; then
   DO_TESTS=""
else
   DO_TESTS="-DLLVM_BUILD_TESTS=OFF -DLLVM_INCLUDE_TESTS=OFF -DCLANG_INCLUDE_TESTS=OFF"
fi

export LLVM_DIR=$AOMP_INSTALL_DIR
AOCC_CMAKE_OPTS="-DLLVM_BUILD_LLVM_DYLIB:STRING=ON -DLLVM_LINK_LLVM_DYLIB=ON -DLLVM_ENABLE_LIBEDIT=OFF -DCMAKE_BUILD_TYPE:STRING=$BUILD_TYPE -DCLANG_DEFAULT_LINKER:STRING=ld.lld -D+++LLVM_BUILD_EXTERNAL_COMPILER_RT=ON -DCOMPILER_RT_BUILD_XRAY=OFF -DLLVM_ENABLE_ASSERTIONS:BOOL=ON -DCMAKE_CXX_FLAGS='-Wno-error=pedantic' -DCMAKE_INSTALL_PREFIX:PATH=$AOMP_INSTALL_DIR"

#FIXME Instead of AOMP_REPOS use $BUILD_DIR after rsync is fixed to replicate all repos when COPYSOURCE is on
EXTERNAL_PROJECTS_OPTS="-DLLVM_EXTERNAL_PROJECTS:STRING=$AOMP_ROCT_COMPONENT_NAME;$AOMP_ROCR_COMPONENT_NAME  \
 -DLLVM_EXTERNAL_${AOMP_ROCT_COMPONENT_NAME^^}_SOURCE_DIR=$AOMP_REPOS/$AOMP_ROCT_REPO_NAME \
 -DLLVM_EXTERNAL_${AOMP_ROCR_COMPONENT_NAME^^}_SOURCE_DIR=$AOMP_REPOS/$AOMP_ROCR_REPO_NAME/src \
"
# Uncomment the following line to build with only LLVM projects.  That is no external projects
#EXTERNAL_PROJECTS_OPTS=""

MYCMAKEOPTS="$AOCC_CMAKE_OPTS -DLLVM_ENABLE_PROJECTS:STRING=clang;lld;compiler-rt -DLLVM_TARGETS_TO_BUILD:STRING=$TARGETS_TO_BUILD $COMPILERS -DLLVM_VERSION_SUFFIX=_${AOMP_COMPILER_NAME}_$AOMP_VERSION_STRING -DBUG_REPORT_URL='$GITROCDEV/aomp' -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF $DO_TESTS $AOMP_ORIGIN_RPATH -DCMAKE_INSTALL_LIBDIR=$AOMP_INSTALL_DIR/lib -DATMI_HSA_INTEROP=on -DATMI_WITH_AOMP=on -DROCM_DIR=$ROCM_DIR -DROCM_VERSION=$ROCM_VERSION -DLLVM_DIR=$LLVM_DIR $EXTERNAL_PROJECTS_OPTS"


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

REPO_BRANCH=$AOMP_PROJECT_REPO_BRANCH
REPO_DIR=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME
checkrepo

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

# Fix the banner to print the AOMP version string. 
if [ $AOMP_STANDALONE_BUILD == 1 ] && [ "$AOMP_PROJECT_REPO_BRANCH" != "master" ] ; then
   cd $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME
   if [ "$AOMP_CHECK_GIT_BRANCH" == 1 ] ; then
      MONO_REPO_ID=`git log | grep -m1 commit | cut -d" " -f2`
   else
      MONO_REPO_ID="build_from_release_source"
   fi
   SOURCEID="Source ID:$AOMP_VERSION_STRING-$MONO_REPO_ID"
   TEMPCLFILE="/tmp/clfile$$.cpp"
   ORIGCLFILE="$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/llvm/lib/Support/CommandLine.cpp"
   if [ $COPYSOURCE ] ; then
      BUILDCLFILE="$BUILD_DIR/$AOMP_PROJECT_REPO_NAME/llvm/lib/Support/CommandLine.cpp"
   else
      BUILDCLFILE=$ORIGCLFILE
   fi

   # FIXME: build WEBSITE string from escaped string "$GITROCDEV/aomp"
   WEBSITE="http\:\/\/github.com\/ROCm-Developer-Tools\/aomp"
   sed "s/LLVM (http:\/\/llvm\.org\/):/${AOMP_COMPILER_NAME}-${AOMP_VERSION_STRING} ($WEBSITE):\\\n $SOURCEID/" $ORIGCLFILE > $TEMPCLFILE
   if [ $? != 0 ] ; then
      echo "ERROR sed command to fix CommandLine.cpp failed."
      exit 1
   fi
fi

#REPO_NAMES="$AOMP_PROJECT_REPO_NAME $AOMP_LIBDEVICE_REPO_NAME $AOMP_HCC_REPO_NAME $AOMP_HIP_REPO_NAME $AOMP_RINFO_REPO_NAME $AOMP_ROCT_REPO_NAME $AOMP_ROCR_REPO_NAME $AOMP_ATMI_REPO_NAME $AOMP_EXTRAS_REPO_NAME $AOMP_COMGR_REPO_NAME $AOMP_FLANG_REPO_NAME"
# Setup for patching and reverse patching 
# For now, just patch the repos for the components that are part of LLVM_EXTERNAL_PROJECTS
REPO_NAMES="$AOMP_PROJECT_REPO_NAME $AOMP_ROCT_REPO_NAME $AOMP_ROCR_REPO_NAME $AOMP_ATMI_REPO_NAME"
patchloc=$thisdir/patches

# Skip synchronization from git repos if nocmake or install are specified
if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/$AOMP_PROJECT_REPO_NAME"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_DIR/build/$AOMP_PROJECT_REPO_NAME
   mkdir -p $BUILD_DIR/build/$AOMP_PROJECT_REPO_NAME

   # Patch the repos in the ExTERNAL PROJECTS
   saveifs=$IFS
   export IFS=" "
   for reponame in $REPO_NAMES
   do
      patchdir=$AOMP_REPOS/$reponame
      patchrepo
   done
   export IFS=$saveifs


   if [ $COPYSOURCE ] ; then 
      #  Copy/rsync the git repos into /tmp for faster compilation
      #FIXME: rsync all repos when all components are part of LLVM_EXTERNAL_PROJECTS
      mkdir -p $BUILD_DIR
      echo
      echo "WARNING!  BUILD_DIR($BUILD_DIR) != AOMP_REPOS($AOMP_REPOS)"
      echo "SO REPLICATING AOMP_REPOS/$AOMP_PROJECT_REPO_NAME  TO: $BUILD_DIR"
      echo rsync -a --exclude ".git" --exclude "CommandLine.cpp" --delete $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME $BUILD_DIR 2>&1 
      rsync -a --exclude ".git" --exclude "CommandLine.cpp" --delete $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME $BUILD_DIR 2>&1 
      echo
   fi

else
   if [ ! -d $BUILD_DIR/build/$AOMP_PROJECT_REPO_NAME ] ; then 
      echo "ERROR: The build directory $BUILD_DIR/build/$AOMP_PROJECT_REPO_NAME does not exist"
      echo "       run $0 without nocmake or install options. " 
      exit 1
   fi
fi

if [ $AOMP_STANDALONE_BUILD == 1 ] && [ "$AOMP_PROJECT_REPO_BRANCH" != "master" ] ; then
   cd $BUILD_DIR/build/$AOMP_PROJECT_REPO_NAME
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
fi

cd $BUILD_DIR/build/$AOMP_PROJECT_REPO_NAME

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake ---- " 
   if [ $COPYSOURCE ] ; then
       echo ${AOMP_CMAKE} -G"Unix Makefiles" $MYCMAKEOPTS  $BUILD_DIR/$AOMP_PROJECT_REPO_NAME/llvm
       ${AOMP_CMAKE} -G"Unix Makefiles" $MYCMAKEOPTS  $BUILD_DIR/$AOMP_PROJECT_REPO_NAME/llvm 2>&1
    else
       echo ${AOMP_CMAKE} -G"Unix Makefiles" $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/llvm
       ${AOMP_CMAKE} -G"Unix Makefiles" $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/llvm 2>&1
   fi
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
   echo " -----Installing to $AOMP_INSTALL_DIR ---- " 
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
   if [ $AOMP_STANDALONE_BUILD == 1 ] ; then 
      echo " "
      echo "------ Linking $AOMP_INSTALL_DIR to $AOMP -------"
      if [ -L $AOMP ] ; then 
         $SUDO rm $AOMP   
      fi
      $SUDO ln -sf $AOMP_INSTALL_DIR $AOMP   
   fi
 
   # Remove all the patches 
   saveifs=$IFS
   export IFS=" "
   for reponame in $REPO_NAMES
   do
      patchdir=$AOMP_REPOS/$reponame
      removepatch
   done
   export IFS=$saveifs

   # add executables forgot by make install but needed for testing
   echo add executables forgot by make install but needed for testing
   $SUDO cp -p $BUILD_DIR/build/$AOMP_PROJECT_REPO_NAME/bin/llvm-lit $AOMP/bin/llvm-lit
   $SUDO cp -p $BUILD_DIR/build/$AOMP_PROJECT_REPO_NAME/bin/FileCheck $AOMP/bin/FileCheck
   $SUDO cp -p $BUILD_DIR/build/$AOMP_PROJECT_REPO_NAME/bin/count $AOMP/bin/count
   $SUDO cp -p $BUILD_DIR/build/$AOMP_PROJECT_REPO_NAME/bin/not $AOMP/bin/not
   $SUDO cp -p $BUILD_DIR/build/$AOMP_PROJECT_REPO_NAME/bin/yaml-bench $AOMP/bin/yaml-bench
   echo
   echo "SUCCESSFUL INSTALL to $AOMP_INSTALL_DIR with link to $AOMP"
   echo
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo 
fi
