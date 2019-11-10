#!/bin/bash
#
#  build_atmi.sh:  Script to build the AOMP atmi libraries and debug libraries.  
#
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
[ ! -L "$0" ] || thisdir=$(getdname `readlink "$0"`)
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_ATMI=${INSTALL_ATMI:-$AOMP_INSTALL_DIR}

REPO_BRANCH=$AOMP_ATMI_REPO_BRANCH
REPO_DIR=$AOMP_REPOS/$AOMP_ATMI_REPO_NAME
checkrepo

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds release and debug versions of ATMI libraries."
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_ATMI_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/atmi"
  echo "    and:                   $BUILD_AOMP/build/atmi_debug"
  echo " It installs in:           $INSTALL_ATMI"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_atmi.sh                   cmake, make , NO Install "
  echo "  ./build_atmi.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_atmi.sh install           NO Cmake, make , INSTALL"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/$AOMP_ATMI_REPO_NAME ] ; then 
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_ATMI_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_ATMI_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_ATMI
   $SUDO touch $INSTALL_ATMI/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_ATMI"
      exit 1
   fi
   $SUDO rm $INSTALL_ATMI/testfile
fi

export LLVM_DIR=$AOMP_INSTALL_DIR
if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then 

   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_atmi"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   BUILDTYPE="Release"
   echo rm -rf $BUILD_AOMP/build/atmi
   rm -rf $BUILD_AOMP/build/atmi
   MYCMAKEOPTS="-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_RPATH=$AOMP_INSTALL_DIR/lib -DCMAKE_BUILD_TYPE=$BUILDTYPE -DATMI_HSA_INTEROP=on -DATMI_WITH_AOMP=on -DCMAKE_INSTALL_PREFIX=$INSTALL_ATMI -DROCM_DIR=$ROCM_DIR -DROCM_VERSION=$ROCM_VERSION -DLLVM_DIR=$LLVM_DIR"
   mkdir -p $BUILD_AOMP/build/atmi
   cd $BUILD_AOMP/build/atmi
   echo
   echo " -----Running atmi cmake ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_ATMI_REPO_NAME/src
   ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_ATMI_REPO_NAME/src
   if [ $? != 0 ] ; then 
      echo "ERROR atmi cmake failed. cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi

   BUILDTYPE="Debug"
   echo rm -rf $BUILD_AOMP/build/atmi_debug
   rm -rf $BUILD_AOMP/build/atmi_debug
   MYCMAKEOPTS="-DCMAKE_FIND_DEBUG_MODE=ON -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_RPATH=$AOMP_INSTALL_DIR/lib-debug:$AOMP_INSTALL_DIR/lib -DCMAKE_BUILD_TYPE=$BUILDTYPE -DATMI_HSA_INTEROP=on -DATMI_WITH_AOMP=on -DCMAKE_INSTALL_PREFIX=$INSTALL_ATMI -DROCM_DIR=$ROCM_DIR -DROCM_VERSION=$ROCM_VERSION -DLLVM_DIR=$LLVM_DIR"

   mkdir -p $BUILD_AOMP/build/atmi_debug
   cd $BUILD_AOMP/build/atmi_debug
   echo
   echo " -----Running atmi cmake for debug ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS $AOMP_REPOS/$AOMP_ATMI_REPO_NAME/src
   ${AOMP_CMAKE} $MYCMAKEOPTS $AOMP_REPOS/$AOMP_ATMI_REPO_NAME/src
   if [ $? != 0 ] ; then 
      echo "ERROR atmi debug cmake failed. cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi
fi

cd $BUILD_AOMP/build/atmi
echo
echo " -----Running make for atmi ---- " 
make -j $NUM_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $NUM_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/atmi"
      echo "  make"
      exit 1
fi

cd $BUILD_AOMP/build/atmi_debug
echo
echo " -----Running make for lib-debug ---- " 
make -j $NUM_THREADS
if [ $? != 0 ] ; then 
      echo "ERROR make -j $NUM_THREADS failed"
      exit 1
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/atmi
      echo " -----Installing to $INSTALL_ATMI/lib ----- " 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
      cd $BUILD_AOMP/build/atmi_debug
      echo " -----Installing to $INSTALL_ATMI/lib-debug ---- " 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
fi
