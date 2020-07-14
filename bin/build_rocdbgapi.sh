#!/bin/bash
#
#  build_rocdbgapi.sh:  Script to build ROCdbgapi for AOMP standalone build
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

INSTALL_ROCDBGAPI=${INSTALL_ROCDBGAPI:-$AOMP_INSTALL_DIR}

REPO_DIR=$AOMP_REPOS/$AOMP_DBGAPI_REPO_NAME
REPO_BRANCH=$AOMP_DBGAPI_REPO_BRANCH
checkrepo

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds the ROCM runtime libraries"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_DBGAPI_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/rocdbgapi"
  echo " It installs in:           $INSTALL_ROCDBGAPI"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocdbgapi.sh                   cmake, make , NO Install "
  echo "  ./build_rocdbgapi.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_rocdbgapi.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/$AOMP_DBGAPI_REPO_NAME ] ; then 
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_DBGAPI_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_DBGAPI_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_ROCDBGAPI
   $SUDO touch $INSTALL_ROCDBGAPI/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_ROCDBGAPI"
      exit 1
   fi
   $SUDO rm $INSTALL_ROCDBGAPI/testfile
fi

API_NAME=rocm-dbgapi
PROJ_NAME=$API_NAME
LIB_NAME=lib${API_NAME}.so
export API_NAME PROJ_NAME LIB_NAME

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then 
   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_rocdbgapi"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   BUILDTYPE="Release"
   echo rm -rf $BUILD_AOMP/build/rocdbgapi
   rm -rf $BUILD_AOMP/build/rocdbgapi
   MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_VERBOSE_MAKEFILE=1 \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_ROCDBGAPI \
	-DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR;$INSTALL_ROCDBGAPI/include \
         $AOMP_ORIGIN_RPATH"
   mkdir -p $BUILD_AOMP/build/rocdbgapi
   cd $BUILD_AOMP/build/rocdbgapi
   echo " -----Running rocdbgapi cmake ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_DBGAPI_REPO_NAME
   ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_DBGAPI_REPO_NAME
   if [ $? != 0 ] ; then 
      echo "ERROR rocdbgapi cmake failed. cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi
fi

cd $BUILD_AOMP/build/rocdbgapi
echo
echo " -----Running make for rocdbgapi ---- " 
echo make -j $NUM_THREADS
make -j $NUM_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $NUM_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/rocdbgapi"
      echo "  make"
      exit 1
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/rocdbgapi
      echo " -----Installing to $INSTALL_ROCDBGAPI/lib ----- " 
      echo $SUDO make install 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
        echo
else
   echo
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo
fi
