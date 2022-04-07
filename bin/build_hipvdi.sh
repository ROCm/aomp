#!/bin/bash
#
#  File: build_hipvdi.sh
#        Build the hipvdi runtimes 
#        The install option will install components into the aomp installation. 
#
# MIT License
#
# Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
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
[ ! -L "$0" ] || thisdir=$(getdname `readlink "$0"`)
. $thisdir/aomp_common_vars
# --- end standard header ----

HIPVDI_REPO_DIR=$AOMP_REPOS/$AOMP_HIPVDI_REPO_NAME

BUILD_DIR=${BUILD_AOMP}

BUILDTYPE="Release"

REPO_DIR=$AOMP_REPOS/$AOMP_HIPVDI_REPO_NAME

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_hipvdi.sh                   cmake, make, NO Install "
  echo "  ./build_hipvdi.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_hipvdi.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d $HIPVDI_REPO_DIR ] ; then
   echo "ERROR:  Missing repository $HIPVDI_REPO_DIR/"
   exit 1
fi

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

patchrepo $AOMP_REPOS/$AOMP_HIPVDI_REPO_NAME

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then

  if [ -d "$BUILD_DIR/build/hipvdi" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_DIR/build/hipvdi
     rm -rf $BUILD_DIR/build/hipvdi
  fi
  ROCclr_BUILD_DIR=$AOMP_REPOS/build/vdi
  ROCclr_ROOT=$AOMP_REPOS/$AOMP_VDI_REPO_NAME
  MYCMAKEOPTS="$AOMP_ORIGIN_RPATH -DCMAKE_BUILD_TYPE=$BUILDTYPE \
 -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR \
 -DHIP_COMPILER=clang \
 -DHIP_PLATFORM=rocclr \
 -DROCM_PATH=$ROCM_DIR \
 -DHSA_PATH=$ROCM_DIR \
 -DCMAKE_MODULE_PATH=$ROCM_DIR/cmake \
 -DROCclr_DIR=$ROCclr_ROOT \
 -D__HIP_ENABLE_PCH=OFF \
 -DLIBROCclr_STATIC_DIR=$ROCclr_BUILD_DIR \
 -DCMAKE_PREFIX_PATH=$ROCclr_BUILD_DIR;$ROCM_DIR/include;$ROCM_DIR;$ROCM_DIR/lib \
 -DCMAKE_CXX_FLAGS=-Wno-ignored-attributes "
 # -DLLVM_INCLUDES=$ROCM_DIR/include "

  echo mkdir -p $BUILD_DIR/build/hipvdi
  mkdir -p $BUILD_DIR/build/hipvdi
  echo cd $BUILD_DIR/build/hipvdi
  cd $BUILD_DIR/build/hipvdi
  echo
  echo " -----Running hipvdi cmake ---- "
  echo ${AOMP_CMAKE} $MYCMAKEOPTS $HIPVDI_REPO_DIR
  export HIP_PLATFORM=ROCclr
  ${AOMP_CMAKE} $MYCMAKEOPTS $HIPVDI_REPO_DIR
  if [ $? != 0 ] ; then
      echo "ERROR hipvdi cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi

fi

cd $BUILD_DIR/build/hipvdi
echo
echo " -----Running make for hipvdi ---- "
make -j $AOMP_JOB_THREADS 
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/hipvdi"
      echo "  make "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install hipvdi component run this command:"
      echo "  $0 install"
      echo
  fi
fi

function edit_installed_hip_file(){
   if [ -f $installed_hip_file_to_edit ] ; then
      # In hipvars.pm HIP_PATH is determined by parent directory of hipcc location.
      # Set ROCM_PATH using HIP_PATH
      $SUDO sed -i -e "s/\"\/opt\/rocm\"/\"\$HIP_PATH\"/" $installed_file_to_edit
      # Set HIP_CLANG_PATH using ROCM_PATH/bin
      $SUDO sed -i -e "s/\"\$ROCM_PATH\/llvm\/bin\"/\"\$ROCM_PATH\/bin\"/" $installed_file_to_edit
    fi
}

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
   cd $BUILD_DIR/build/hipvdi
   echo
   echo " -----Installing to $AOMP_INSTALL_DIR ----- "
   $SUDO make install
   if [ $? != 0 ] ; then
      echo "ERROR make install failed "
      exit 1
   fi
   removepatch $AOMP_REPOS/$AOMP_HIPVDI_REPO_NAME

   # The hip perl scripts have /opt/rocm hardcoded, so fix them after then are installed
   # but only if not installing to rocm.
   if [ $AOMP_INSTALL_DIR != "/opt/rocm/llvm" ] ; then
      SED_INSTALL_DIR=`echo $AOMP_INSTALL_DIR | sed -e 's/\//\\\\\//g' `
      installed_file_to_edit=$AOMP_INSTALL_DIR/bin/hipcc
      $(edit_installed_hip_file)
      installed_file_to_edit=$AOMP_INSTALL_DIR/bin/hipvars.pm
      $(edit_installed_hip_file)
      # nothing to change in hipconfig but in case something is added in future, try to fix it
      installed_file_to_edit=$AOMP_INSTALL_DIR/bin/hipconfig
      $(edit_installed_hip_file)
   fi

fi
