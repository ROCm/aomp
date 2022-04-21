#!/bin/bash
#
#  build_comgr.sh:  Script to build the code object manager for aomp
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

INSTALL_COMGR=${INSTALL_COMGR:-$AOMP_INSTALL_DIR}

REPO_DIR=$AOMP_REPOS/$AOMP_COMGR_REPO_NAME

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds the code object manager"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_COMGR_REPO_NAME/lib/comgr"
  echo " It builds libraries in:   $BUILD_AOMP/build/comgr"
  echo " It installs in:           $INSTALL_COMGR"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_comgr.sh                   cmake, make , NO Install "
  echo "  ./build_comgr.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_comgr.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/$AOMP_COMGR_REPO_NAME ] ; then 
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_COMGR_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_COMGR_REPO_NAME set correctly?"
   exit 1
fi

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

osversion=$(cat /etc/os-release)
#if [ "$AOMP_MAJOR_VERSION" != "12" ] && [[ "$osversion" =~ "Ubuntu 16" ]];  then
  patchrepo $REPO_DIR
#fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then

   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_comgr"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   BUILDTYPE="Release"
   echo $SUDO rm -rf $BUILD_AOMP/build/comgr
   $SUDO rm -rf $BUILD_AOMP/build/comgr
   export LLVM_DIR=$AOMP_INSTALL_DIR
   export Clang_DIR=$AOMP_INSTALL_DIR

   mkdir -p $BUILD_AOMP/build/comgr
   cd $BUILD_AOMP/build/comgr
   echo " -----Running comgr cmake ---- " 

   DEVICELIBS_BUILD_PATH=$AOMP_REPOS/build/AOMP_LIBDEVICE_REPO_NAME
   PACKAGE_ROOT=$AOMP_REPOS/$AOMP_COMGR_REPO_NAME/lib/comgr
   ${AOMP_CMAKE} \
      -DCMAKE_PREFIX_PATH="$AOMP/include/amd_comgr;$AOMP/lib/cmake;$DEVICELIBS_BUILD_PATH;$PACKAGE_ROOT" \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_COMGR \
      -DCMAKE_BUILD_TYPE=$BUILDTYPE \
      -DBUILD_TESTING=OFF \
      $AOMP_ORIGIN_RPATH \
      -DROCM_DIR=$AOMP_INSTALL_DIR  \
      -DLLVM_DIR=$AOMP_INSTALL_DIR \
      -DClang_DIR=$AOMP_INSTALL_DIR \
      $AOMP_REPOS/$AOMP_COMGR_REPO_NAME/lib/comgr

   if [ $? != 0 ] ; then 
      echo "ERROR comgr cmake failed. cmake flags"
      exit 1
   fi

fi

cd $BUILD_AOMP/build/comgr
echo
echo " -----Running make for comgr ---- " 
make -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/comgr"
      echo "  make"
      exit 1
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/comgr
      echo " -----Installing to $INSTALL_COMGR/lib ----- " 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
      # amd_comgr.h is now in amd_comgr/amd_comgr.h, so remove depracated file
      [ -f $INSTALL_COMGR/include/amd_comgr.h ] && rm $INSTALL_COMGR/include/amd_comgr.h
      if [ "$AOMP_MAJOR_VERSION" != "12" ] && [[ "$osversion" =~ "Ubuntu 16" ]]; then
        removepatch $REPO_DIR
      fi
fi
