#!/bin/bash
# 
#  build_hcc.sh: Script to build hcc for the AOMP compiler. 
#                hcc is only needed to build hip-clang
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

INSTALL_HCC=${INSTALL_HCC:-$AOMP_INSTALL_DIR/hcc}

if [ "$AOMP_PROC" == "ppc64le" ] ; then 
   GCCMIN=8
   TARGETS_TO_BUILD="AMDGPU;PowerPC"
else
   GCCMIN=5
   if [ "$AOMP_PROC" == "aarch64" ] ; then 
      TARGETS_TO_BUILD="AMDGPU;AArch64"
   else
      TARGETS_TO_BUILD="AMDGPU;X86"
   fi
fi

function getgccmin(){
   _loc=`which gcc`
   [ "$_loc" == "" ] && return
   gccver=`$_loc --version | grep gcc | cut -d")" -f2 | cut -d"." -f1`
   [ $gccver -lt $GCCMIN ] && _loc=`which gcc-$GCCMIN`
   echo $_loc
}
function getgxxmin(){
   _loc=`which g++`
   [ "$_loc" == "" ] && return
   gxxver=`$_loc --version | grep g++ | cut -d")" -f2 | cut -d"." -f1`
   [ $gxxver -lt $GCCMIN ] && _loc=`which g++-$GCCMIN`
   echo $_loc
}
GCCLOC=$(getgccmin)
GXXLOC=$(getgxxmin)

if [ "$GCCLOC" == "" ] ; then
   echo "ERROR: NO ADEQUATE gcc"
   echo "       Please install at least gcc-$GCCMIN"
   exit 1
fi
if [ "$GXXLOC" == "" ] ; then
   echo "ERROR: NO ADEQUATE g++"
   echo "       Please install at least g++-$GCCMIN"
   exit 1
fi
COMPILERS="-DCMAKE_C_COMPILER=$GCCLOC -DCMAKE_CXX_COMPILER=$GXXLOC"

# When building from release source (no git), turn off tests which are not part of release source
if [ "$AOMP_CHECK_GIT_BRANCH" == 1 ] ; then
   DO_TESTS=""
else
   DO_TESTS="-DLLVM_BUILD_TESTS=OFF -DLLVM_INCLUDE_TESTS=OFF -DCLANG_INCLUDE_TESTS=OFF"
fi


GFXSEMICOLONS=`echo $GFXLIST | tr ' ' ';' `
MYCMAKEOPTS="-DROCM_ROOT=$AOMP_INSTALL_DIR -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_RPATH=$AOMP_INSTALL_DIR/lib:$AOMP_INSTALL_DIR/hcc/lib -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_HCC $COMPILERS -DHSA_AMDGPU_GPU_TARGET=$GFXSEMICOLONS -DHSA_HEADER_DIR=$AOMP_INSTALL_DIR/hsa/include -DHSA_LIBRARY_DIR=$AOMP_INSTALL_DIR/hsa/lib -DCLANG_ANALYZER_ENABLE_Z3_SOLVER=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DCMAKE_INSTALL_LIBDIR=$AOMP_INSTALL_DIR/hcc/lib $DO_TESTS"

# -DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD 
#  For now build hcc with ROCDL. because using the AOMP libdevice 
#  with the following option would force build_hcc.sh to be after build_libdevice.sh
#  And since build_hip.sh must be after build_hcc.sh, build_hip.sh would be almost last.
# -DHCC_INTEGRATE_ROCDL=OFF -DROCM_DEVICE_LIB_DIR=$AOMP_INSTALL_DIR/lib/libdevice
# 

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

REPO_BRANCH=$AOMP_HCC_REPO_BRANCH
REPO_DIR=$AOMP_REPOS/$AOMP_HCC_REPO_NAME
checkrepo

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_HCC
   $SUDO touch $INSTALL_HCC/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_HCC"
      exit 1
   fi
   $SUDO rm $INSTALL_HCC/testfile
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   #  When build_hcc.sh is run with no args, start fresh and clean out the build directory.
   echo 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/$AOMP_HCC_REPO_NAME"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_DIR/build/$AOMP_HCC_REPO_NAME
   mkdir -p $BUILD_DIR/build/$AOMP_HCC_REPO_NAME

   if [ $COPYSOURCE ] ; then 
      # IF REQUESTED (NOT DEFAULT), establish the sources in a different location than the git repos
      # Normally this is done with rsync, but for hcc we cannot depend on rsync
      mkdir -p $BUILD_DIR
      echo
      echo "WARNING!  BUILD_DIR($BUILD_DIR) != AOMP_REPOS($AOMP_REPOS)"
      echo "   SO CLONING REPO $AOMP_HCC_REPO_NAME TO: $BUILD_DIR/$AOMP_HCC_REPO_NAME"
      echo "   ANY $AOMP_HCC_REPO_NAME REPO IN $AOMP_REPOS IS IGNORED!"
      echo "   THIS IS DIFFERENT THAN OTHER BUILD SCRIPTS BECAUSE rsync OF SUBMODULES DOES NOT WORK"
      #rsync -a --delete $AOMP_REPOS/$AOMP_HCC_REPO_NAME $BUILD_DIR 2>&1 
      if [ -d $BUILD_DIR/hcc ] ; then 
         echo "   Checking for updates to repo $AOMP_HCC_REPO_NAME branch $AOMP_HCC_REPO_BRANCH"
         echo "   cd $BUILD_DIR/hcc"
         cd $BUILD_DIR/hcc
         git checkout $AOMP_HCC_REPO_BRANCH
         echo "   git submodule update"
         git submodule update
         if [ $? != 0 ] ; then
            echo "ERROR: 'git submodule update' failed. Consider deleting the $AOMP_HCC_REPO_NAME repo"
            echo "       at $BUILD_DIR/hcc and letting this build script clone it"
            exit 1
         fi
      else
         echo "   Cloning repo $AOMP_HCC_REPO_NAME branch $AOMP_HCC_REPO_BRANCH"
         echo "   cd $BUILD_DIR"
         cd $BUILD_DIR
         echo "   git clone -b $AOMP_HCC_REPO_BRANCH --recursive $GITROC/$AOMP_HCC_REPO_NAME.git"
         git clone -b $AOMP_HCC_REPO_BRANCH --recursive $GITROC/$AOMP_HCC_REPO_NAME.git
         cd hcc
      fi
      if [ "$AOMP_PROC" == "ppc64le" ] ; then
         patchfile=$thisdir/patches/hcc_ppc_fp16.patch
         patchdir=$BUILD_DIR/hcc
         patchrepo
      fi

      echo "   git status"
      git status
      echo
   fi

else
# Skip synchronization from git repos if nocmake or install are specified
# But ensure the build repo as been created from previous run with no args. 
   if [ ! -d $BUILD_DIR/build/$AOMP_HCC_REPO_NAME ] ; then 
      echo "ERROR: The build directory $BUILD_DIR/build/$AOMP_HCC_REPO_NAME does not exist"
      echo "       run $0 without nocmake or install options. " 
      exit 1
   fi
fi

cd $BUILD_DIR/build/$AOMP_HCC_REPO_NAME

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   # run cmake on fresh starts only, that is no args
   echo
   echo " -----Running cmake ---- " 
   if [ $COPYSOURCE ] ; then
      echo ${AOMP_CMAKE} $MYCMAKEOPTS  $BUILD_DIR/$AOMP_HCC_REPO_NAME
      ${AOMP_CMAKE} $MYCMAKEOPTS  $BUILD_DIR/$AOMP_HCC_REPO_NAME 2>&1
   else
      echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_HCC_REPO_NAME
      ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_HCC_REPO_NAME 2>&1
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
   echo "ERROR make -j $NUM_THREADS FAILED!"
   exit 1
fi

#  The build script always runs make
if [ "$1" == "install" ] ; then
   echo
   echo " -----Installing to $INSTALL_HCC ---- " 
   $SUDO make install 
   if [ $? != 0 ] ; then 
      echo "ERROR make install failed "
      exit 1
   fi
   echo
   echo "SUCCESSFUL INSTALL to $INSTALL_HCC to $AOMP_INSTALL_DIR/hcc"
   echo
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install hcc into $AOMP/hcc"
   echo 
fi

