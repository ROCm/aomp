#!/bin/bash
#
#  build_roct.sh:  Script to build the ROCt thunk libraries.
#
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

INSTALL_ROCT=${INSTALL_ROCT:-$AOMP_INSTALL_DIR}

REPO_DIR=$AOMP_REPOS/$AOMP_ROCT_REPO_NAME
REPO_BRANCH=$AOMP_ROCT_REPO_BRANCH
checkrepo

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

patchfile=$thisdir/patches/roct_ppc.patch
patchdir=$AOMP_REPOS/$AOMP_ROCT_REPO_NAME
patchrepo

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then 

   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_roct"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   BUILDTYPE="Release"
   echo $SUDO rm -rf $BUILD_AOMP/build/roct
   $SUDO rm -rf $BUILD_AOMP/build/roct
   MYCMAKEOPTS="-DCMAKE_INSTALL_PREFIX=$INSTALL_ROCT -DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_RPATH=$AOMP_INSTALL_DIR/lib -DCMAKE_INSTALL_LIBDIR=$AOMP_INSTALL_DIR/lib"
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
make -j $NUM_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $NUM_THREADS  FAILED"
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
      $SUDO make install-dev
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
fi
