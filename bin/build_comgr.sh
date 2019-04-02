#!/bin/bash
# 
#  build_comgr.sh:  Script to build the comgr component of the AOMP compiler. 
#
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

INSTALL_COMGR=${INSTALL_COMGR:-$AOMP_INSTALL_DIR}

GCC=`which gcc`
GCPLUSCPLUS=`which g++`
if [ "$AOMP_PROC" == "ppc64le" ] ; then
   COMPILERS="-DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7"
else
   COMPILERS="-DCMAKE_C_COMPILER=$GCC -DCMAKE_CXX_COMPILER=$GCPLUSCPLUS"
fi
MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_COMGR -DLLVM_DIR=$AOMP_INSTALL_DIR/lib/cmake/llvm -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_TARGETS_TO_BUILD=AMDGPU $COMPILERS -DLLVM_VERSION_SUFFIX=_AOMP_Version_$AOMP_VERSION_STRING -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_RPATH=$AOMP_INSTALL_DIR/lib"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  help_build_aomp
fi

REPO_BRANCH=$AOMP_COMGR_REPO_BRANCH
REPO_DIR=$AOMP_REPOS/$AOMP_COMGR_REPO_NAME
checkrepo

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_COMGR
   $SUDO touch $INSTALL_COMGR/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_COMGR"
      exit 1
   fi
   $SUDO rm $INSTALL_COMGR/testfile
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
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/$AOMP_COMGR_REPO_NAME"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_DIR/build/$AOMP_COMGR_REPO_NAME
   mkdir -p $BUILD_DIR/build/$AOMP_COMGR_REPO_NAME

   if [ $COPYSOURCE ] ; then 
      #  Copy/rsync the git repos into /tmp for faster compilation
      mkdir -p $BUILD_DIR
      echo
      echo "WARNING!  BUILD_DIR($BUILD_DIR) != AOMP_REPOS($AOMP_REPOS)"
      echo "SO RSYNCING AOMP_REPOS TO: $BUILD_DIR"
      echo
      echo rsync -a --exclude ".git" --delete $AOMP_REPOS/$AOMP_COMGR_REPO_NAME $BUILD_DIR
      rsync -a --exclude ".git" --delete $AOMP_REPOS/$AOMP_COMGR_REPO_NAME $BUILD_DIR 2>&1
   fi
else
   if [ ! -d $BUILD_DIR/build/$AOMP_COMGR_REPO_NAME ] ; then 
      echo "ERROR: The build directory $BUILD_DIR/build/$AOMP_COMGR_REPO_NAME does not exist"
      echo "       run $0 without nocmake or install options. " 
      exit 1
   fi
fi

cd $BUILD_DIR/build/$AOMP_COMGR_REPO_NAME

#  Need llvm-config to come from previous LLVM build
export PATH=$AOMP_INSTALL_DIR/bin:$PATH

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake ---- " 
   echo cmake $MYCMAKEOPTS  $BUILD_DIR/$AOMP_COMGR_REPO_NAME/lib/comgr
   cmake $MYCMAKEOPTS  $BUILD_DIR/$AOMP_COMGR_REPO_NAME/lib/comgr 2>&1
   if [ $? != 0 ] ; then 
      echo "ERROR cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi
fi

echo
echo " -----Running make ---- " 
echo make amd_comgr -j $NUM_THREADS 
make amd_comgr -j $NUM_THREADS 
if [ $? != 0 ] ; then 
   echo "ERROR make -j $NUM_THREADS failed"
   exit 1
fi

if [ "$1" == "install" ] ; then
   echo " -----Installing to $INSTALL_COMGR ---- " 
   echo "make install/strip "
   $SUDO make install/strip 
   if [ $? != 0 ] ; then 
      echo "ERROR make install/strip failed "
      exit 1
   fi
   echo "SUCCESSFUL INSTALL to $INSTALL_COMGR with link to $AOMP"
   echo
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo 
fi
