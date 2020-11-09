#!/bin/bash
#
#  build_rocr.sh:  Script to build the rocm runtime and install into the 
#                  aomp compiler installation
#                  Requires that "build_roct.sh install" be installed first
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
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_ROCM=${INSTALL_ROCM:-$AOMP_INSTALL_DIR}

REPO_DIR=$AOMP_REPOS/$AOMP_ROCR_REPO_NAME
REPO_BRANCH=$AOMP_ROCR_REPO_BRANCH
checkrepo

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds the ROCM runtime libraries"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_ROCR_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/rocr"
  echo " It installs in:           $INSTALL_ROCM"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocr.sh                   cmake, make , NO Install "
  echo "  ./build_rocr.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_rocr.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/$AOMP_ROCR_REPO_NAME ] ; then 
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_ROCR_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_ROCR_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_ROCM
   $SUDO touch $INSTALL_ROCM/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_ROCM"
      exit 1
   fi
   $SUDO rm $INSTALL_ROCM/testfile
fi

patchrepo $AOMP_REPOS/$AOMP_ROCR_REPO_NAME

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then 

   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_rocr"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   BUILDTYPE="Release"
   echo rm -rf $BUILD_AOMP/build/rocr
   rm -rf $BUILD_AOMP/build/rocr
   MYCMAKEOPTS="-DCMAKE_INSTALL_PREFIX=$INSTALL_ROCM -DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_PREFIX_PATH=$ROCM_DIR -DIMAGE_SUPPORT=OFF $AOMP_ORIGIN_RPATH"
   mkdir -p $BUILD_AOMP/build/rocr
   cd $BUILD_AOMP/build/rocr
   echo " -----Running rocr cmake ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_ROCR_REPO_NAME/src
   ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_ROCR_REPO_NAME/src
   if [ $? != 0 ] ; then 
      echo "ERROR rocr cmake failed. cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi

fi

cd $BUILD_AOMP/build/rocr
echo
echo " -----Running make for rocr ---- " 
echo make -j $NUM_THREADS
make -j $NUM_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $NUM_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/rocr"
      echo "  make"
      exit 1
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/rocr
      echo " -----Installing to $INSTALL_ROCM/lib ----- " 
      echo $SUDO make install 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
      removepatch $AOMP_REPOS/$AOMP_ROCR_REPO_NAME
      # Remove hsa directory from install to ensure it is not used
      if [ -d $INSTALL_ROCM/hsa ] ; then
         echo rm -rf $INSTALL_ROCM/hsa
         rm -rf $INSTALL_ROCM/hsa
      fi
fi
