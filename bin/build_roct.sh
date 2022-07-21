#!/bin/bash
#
#  build_roct.sh:  Script to build the ROCt thunk libraries.
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_ROCT=${INSTALL_ROCT:-$AOMP_INSTALL_DIR}

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds the ROCt thunk libraries"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_ROCT_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/roct"
  echo " It installs in:           $INSTALL_ROCT"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_roct.sh                   cmake, make , NO Install "
  echo "  ./build_roct.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_roct.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/$AOMP_ROCT_REPO_NAME ] ; then 
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_ROCT_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_ROCT_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_ROCT
   $SUDO touch $INSTALL_ROCT/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_ROCT"
      exit 1
   fi
   $SUDO rm $INSTALL_ROCT/testfile
fi

patchrepo $AOMP_REPOS/$AOMP_ROCT_REPO_NAME

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then 

   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_roct"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   BUILDTYPE="Release"
   echo $SUDO rm -rf $BUILD_AOMP/build/roct
   $SUDO rm -rf $BUILD_AOMP/build/roct
   MYCMAKEOPTS="-DCMAKE_INSTALL_PREFIX=$INSTALL_ROCT -DCMAKE_BUILD_TYPE=$BUILDTYPE $AOMP_ORIGIN_RPATH -DCMAKE_INSTALL_LIBDIR=$AOMP_INSTALL_DIR/lib -DBUILD_SHARED_LIBS=ON"
   mkdir -p $BUILD_AOMP/build/roct
   cd $BUILD_AOMP/build/roct
   echo " -----Running roct cmake ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_ROCT_REPO_NAME
   ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_ROCT_REPO_NAME
   if [ $? != 0 ] ; then 
      echo "ERROR roct cmake failed. cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi

fi

cd $BUILD_AOMP/build/roct
echo
echo " -----Running make for roct ---- " 
make -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/roct"
      echo "  make"
      exit 1
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/roct
      echo " -----Installing to $INSTALL_ROCT/lib ----- " 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
      removepatch $AOMP_REPOS/$AOMP_ROCT_REPO_NAME
fi
