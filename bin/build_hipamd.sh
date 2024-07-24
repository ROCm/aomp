#!/bin/bash
#
#  File: build_hipamd.sh
#        Build hip from hipamd, hip, ROCclr, and ROCm-OpenCL-Runtime repos
#        The install option will install components into the aomp installation. 
#
# MIT License
#
# Copyright (c) 2021 Advanced Micro Devices, Inc. All Rights Reserved.
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
# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

export HIPAMD_DIR=$AOMP_REPOS/clr
export HIP_DIR=$AOMP_REPOS/hip
export ROCclr_DIR=$AOMP_REPOS/clr/rocclr
export OPENCL_DIR=$AOMP_REPOS/clr/opencl
[[ ! -d $HIPAMD_DIR ]] && echo "ERROR:  Missing $HIPAMD_DIR" && exit 1
[[ ! -d $HIP_DIR ]]    && echo "ERROR:  Missing $HIP_DIR"    && exit 1
[[ ! -d $ROCclr_DIR ]] && echo "ERROR:  Missing $ROCclr_DIR" && exit 1
[[ ! -d $OPENCL_DIR ]] && echo "ERROR:  Missing $OPENCL_DIR" && exit 1

export HSA_PATH=$AOMP_INSTALL_DIR
export ROCM_PATH=$AOMP_INSTALL_DIR
export HIP_CLANG_PATH=$AOMP_INSTALL_DIR/bin
export DEVICE_LIB_PATH=$AOMP_INSTALL_DIR/lib

HIP_PATH=$AOMP_INSTALL_DIR
BUILD_DIR=${BUILD_AOMP}
BUILDTYPE="Release"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_hipamd.sh                   cmake, make, NO Install "
  echo "  ./build_hipamd.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_hipamd.sh install           NO Cmake, make install "
  echo " "
  exit
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

patchrepo $AOMP_REPOS/hipamd

if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
  LDFLAGS="-fuse-ld=lld $ASAN_FLAGS"
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then

  if [ -d "$BUILD_DIR/build/hipamd" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_DIR/build/hipamd
     rm -rf $BUILD_DIR/build/hipamd
  fi

  export ROCM_RPATH="$AOMP_ORIGIN_RPATH_LIST"
  MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILDTYPE \
 -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR \
 -DHIP_COMMON_DIR=$HIP_DIR \
 -DHIP_PLATFORM=amd \
 -DHIP_COMPILER=clang \
 -DCMAKE_HIP_ARCHITECTURES=OFF \
 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=ON \
 -DHIPCC_BIN_DIR=$AOMP_REPOS/build/hipcc \
 -DROCM_PATH=$ROCM_PATH"

  if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
     ASAN_FLAGS="$ASAN_FLAGS -I$SANITIZER_COMGR_INCLUDE_PATH -Wno-error=deprecated-declarations"
     ASAN_CMAKE_OPTS="$MYCMAKEOPTS $AOMP_ASAN_ORIGIN_RPATH -DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR/lib/asan/cmake;$AOMP_INSTALL_DIR -DCMAKE_INSTALL_LIBDIR=lib/asan -DCMAKE_C_COMPILER=$LLVM_INSTALL_LOC/bin/clang -DCMAKE_CXX_COMPILER=$LLVM_INSTALL_LOC/bin/clang++"
  fi

  MYCMAKEOPTS="$MYCMAKEOPTS $AOMP_ORIGIN_RPATH -DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_CXX_FLAGS=-I$AOMP_include/amd_comgr -DCMAKE_CXX_FLAGS=-Wno-error=deprecated-declarations -DCMAKE_C_FLAGS=-Wno-error=deprecated-declarations"

  # If this machine does not have an actvie amd GPU, tell hipamd
  # to use first in GFXLIST or gfx90a if no GFXLIST
  if [ -f $AOMP/bin/amdgpu-arch ] ; then
     $AOMP/bin/amdgpu-arch >/dev/null
     if [ $? != 0 ] ; then
	if [ ! -z "$GFXLIST" ] ; then
	   amdgpu=`echo $GFXLIST | cut -d" " -f1`
        else
           amdgpu=gfx90a
	fi
        MYCMAKEOPTS+=" -DOFFLOAD_ARCH_STR=$amdgpu"
     fi
  fi

  echo mkdir -p $BUILD_DIR/build/hipamd
  mkdir -p $BUILD_DIR/build/hipamd
  echo cd $BUILD_DIR/build/hipamd
  cd $BUILD_DIR/build/hipamd
  echo
  echo " -----Running hipamd cmake ---- "
  echo ${AOMP_CMAKE} $MYCMAKEOPTS $HIPAMD_DIR
  ${AOMP_CMAKE} $MYCMAKEOPTS $HIPAMD_DIR
  if [ $? != 0 ] ; then
      echo "ERROR hipamd cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi

  if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
     echo mkdir -p $BUILD_DIR/build/hipamd/asan
     mkdir -p $BUILD_DIR/build/hipamd/asan
     echo cd $BUILD_DIR/build/hipamd/asan
     cd $BUILD_DIR/build/hipamd/asan
     echo
     echo " -----Running hipamd-asan cmake -----"
     echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DOFFLOAD_ARCH_STR=$amdgpu -DCMAKE_CXX_FLAGS="$ASAN_FLAGS" -DCMAKE_CXX_FLAGS="$ASAN_FLAGS" $HIPAMD_DIR
     ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DOFFLOAD_ARCH_STR="$amdgpu" -DCMAKE_CXX_FLAGS="$ASAN_FLAGS" -DCMAKE_CXX_FLAGS="$ASAN_FLAGS" $HIPAMD_DIR
     if [ $? != 0 ] ; then
        echo "ERROR hipamd-asan cmake failed. Cmake flags"
        echo "      $ASAN_CMAKE_OPTS"
        exit 1
     fi
  fi
fi

cd $BUILD_DIR/build/hipamd

echo
echo " -----Running make for hipamd ---- "
make -j $AOMP_JOB_THREADS 
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/hipamd"
      echo "  make "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install hipamd component run this command:"
      echo "  $0 install"
      echo
  fi
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
   cd $BUILD_DIR/build/hipamd/asan
   echo
   echo " -----Running make for hipamd-asan ----- "
   make -j $AOMP_JOB_THREADS
   if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS FAILED"
      echo "To restart:"
      echo "  cd restart:"
      echo "  make "
      exit 1
   else
      if [ "$1" != "install" ] ; then
         echo
         echo " BUILD COMPLETE! To install hipamd-asan component run this command:"
         echo " $0 install"
         echo
      fi
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
   cd $BUILD_DIR/build/hipamd
   echo
   echo " -----Installing to $AOMP_INSTALL_DIR ----- "
   $SUDO make install
   if [ $? != 0 ] ; then
      echo "ERROR make install failed "
      exit 1
   fi

   if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
      cd $BUILD_DIR/build/hipamd/asan
      echo
      echo " -----Installing to $AOMP_INSTALL_DIR/lib/asan"
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
   fi
   removepatch $AOMP_REPOS/hipamd

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
