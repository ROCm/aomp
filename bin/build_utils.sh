#!/bin/bash
#
#  File: build_utils.sh
#        Build the AOMP utilities
#        The install option will install components into the aomp installation. 
#
# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

INSTALL_UTILS=${INSTALL_UTILS:-$AOMP_INSTALL_DIR}

UTILS_DIR=${AOMP_REPOS}/$AOMP_REPO_NAME/utils

MYCMAKEOPTS="-DCMAKE_INSTALL_PREFIX=$INSTALL_UTILS -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_RPATH=$AOMP_INSTALL_DIR/lib"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_utils.sh                   cmake, make, NO Install "
  echo "  ./build_utils.sh nocmake           NO Cmake, make, NO install "
  echo "  ./build_utils.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d $UTILS_DIR ] ; then
   echo "ERROR:  The directory $UTILS_DIR was not found"
   exit 1
fi

if [ ! -f $AOMP/bin/clang ] ; then
   echo "ERROR:  Missing file $AOMP/bin/clang"
   echo "        Build the AOMP llvm compiler in $AOMP first"
   echo "        This is needed to build the utilities"
   echo " "
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then
   $SUDO mkdir -p $INSTALL_UTILS
   $SUDO touch $INSTALL_UTILS/testfile
   if [ $? != 0 ] ; then
      echo "ERROR: No update access to $INSTALL_UTILS"
      exit 1
   fi
   $SUDO rm $INSTALL_UTILS/testfile
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then

  if [ -d "$BUILD_DIR/build/utils" ] ; then
     echo
     echo " FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo " rm -rf $BUILD_DIR/build/utils"
     rm -rf $BUILD_DIR/build/utils
  fi

  mkdir -p $BUILD_DIR/build/utils
  cd $BUILD_DIR/build/utils
  echo 
  echo " ---- Running cmake at $BUILD_DIR/build/utils ---- "
  echo cmake $MYCMAKEOPTS $UTILS_DIR
  cmake $MYCMAKEOPTS $UTILS_DIR
  if [ $? != 0 ] ; then
      echo "ERROR utils cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi
fi

echo
echo " ---- Running make at $BUILD_DIR/build/utils ---- "
cd $BUILD_DIR/build/utils
make
if [ $? != 0 ] ; then
      echo
      echo "ERROR: make FAILED"
      echo
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install in $INSTALL_UTILS/bin run this command:"
      echo "  $0 install"
      echo
  fi
fi

if [ "$1" == "install" ] ; then
      echo
      echo " ---- Installing to $INSTALL_UTILS/bin ----- "
      cd $BUILD_DIR/build/utils
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
      echo
      echo " INSTALL COMPLETE!"
      echo
fi
