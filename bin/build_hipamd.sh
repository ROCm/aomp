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
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/aomp_common_vars"
# --- end standard header ----

export HIPAMD_DIR=$AOMP_REPOS/${AOMP_ROCM_SYSTEMS_NAME}/clr
export HIP_DIR=$AOMP_REPOS/${AOMP_ROCM_SYSTEMS_NAME}/hip
export ROCclr_DIR=$AOMP_REPOS/${AOMP_ROCM_SYSTEMS_NAME}/clr/rocclr
export OPENCL_DIR=$AOMP_REPOS/${AOMP_ROCM_SYSTEMS_NAME}/clr/opencl
[[ ! -d $HIPAMD_DIR ]] && echo "ERROR:  Missing $HIPAMD_DIR" && exit 1
[[ ! -d $HIP_DIR ]]    && echo "ERROR:  Missing $HIP_DIR"    && exit 1
[[ ! -d $ROCclr_DIR ]] && echo "ERROR:  Missing $ROCclr_DIR" && exit 1
[[ ! -d $OPENCL_DIR ]] && echo "ERROR:  Missing $OPENCL_DIR" && exit 1

export HSA_PATH=$AOMP_INSTALL_DIR
export ROCM_PATH=$AOMP_INSTALL_DIR
export HIP_CLANG_PATH=$AOMP_INSTALL_DIR/bin
export DEVICE_LIB_PATH=$AOMP_INSTALL_DIR/lib
export LLVM_DIR=$LLVM_INSTALL_LOC

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

check_writable_installdir "$1" "$AOMP_INSTALL_DIR"

patchrepo "$AOMP_REPOS/hipamd"
patchrepo "$AOMP_REPOS/clr"

#if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
  #LDFLAGS=$(shquot '-fuse-ld=lld' "${ASAN_FLAGS[@]}")"
  #export LDFLAGS
#fi

_ompd_src_dir="$LLVM_INSTALL_LOC/share/gdb/python/ompd/src"

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then

  if [ -d "$BUILD_DIR/build/hipamd" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf "$BUILD_DIR/build/hipamd"
     rm -rf "$BUILD_DIR/build/hipamd"
  fi

  declare -a MYCMAKEOPTS

  MYCMAKEOPTS=(-DCMAKE_BUILD_TYPE="$BUILDTYPE"
               -DCMAKE_INSTALL_PREFIX="$AOMP_INSTALL_DIR"
               -DHIP_COMMON_DIR="$HIP_DIR"
               -DHIP_PLATFORM=amd
               -DHIP_COMPILER=clang
               -DCMAKE_HIP_ARCHITECTURES=OFF
               -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=ON
               -DHIPCC_BIN_DIR="$BUILD_DIR/build/hipcc"
               -DROCM_PATH="$ROCM_PATH"
               -DBUILD_ICD=ON)

  # If this machine does not have an actvie amd GPU, tell hipamd
  # to use first in GFXLIST or gfx90a if no GFXLIST
  if [ -f "$LLVM_INSTALL_LOC/bin/amdgpu-arch" ] ; then
     if ! "$LLVM_INSTALL_LOC/bin/amdgpu-arch" >/dev/null; then
        if [ -n "$GFXLIST" ] ; then
           amdgpu=$(echo "$GFXLIST" | cut -d" " -f1)
        else
           amdgpu=gfx90a
	     fi
        MYCMAKEOPTS=("${MYCMAKEOPTS[@]}" "-DOFFLOAD_ARCH_STR=$amdgpu")
     fi
  fi

  if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
     ASAN_FLAGS=("${ASAN_FLAGS[@]}" -I"$SANITIZER_COMGR_INCLUDE_PATH" -Wno-error=deprecated-declarations)
     ASAN_CMAKE_OPTS=("${MYCMAKEOPTS[@]}" "${AOMP_ASAN_ORIGIN_RPATH[@]}"
                      -DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib/asan/cmake;$AOMP_INSTALL_DIR;$HOME/local/openclicdloader;$BUILD_DIR/build/hipamd/opencl/khronos/icd"
                      -DCMAKE_INSTALL_LIBDIR=lib/asan
                      -DCMAKE_C_COMPILER="$LLVM_INSTALL_LOC/bin/clang"
                      -DCMAKE_CXX_COMPILER="$LLVM_INSTALL_LOC/bin/clang++"
                      -DHIP_LLVM_ROOT="$LLVM_INSTALL_LOC")
  fi

  if [ "$AOMP_BUILD_DEBUG" == 1 ]; then
     HIPAMD_DEBUG_CMAKE_OPTS=("${MYCMAKEOPTS[@]}"
                              "${AOMP_DEBUG_ORIGIN_RPATH[@]}"
                              -DCMAKE_BUILD_TYPE=DEBUG
                              -DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR;$HOME/local/openclicdloader;$BUILD_DIR/build/hipamd/opencl/khronos/icd"
                              -DCMAKE_INSTALL_LIBDIR=lib-debug
                              -DCMAKE_C_COMPILER="$LLVM_INSTALL_LOC/bin/clang"
                              -DCMAKE_CXX_COMPILER="$LLVM_INSTALL_LOC/bin/clang++"
                              -DHIP_LLVM_ROOT="$LLVM_INSTALL_LOC")
  fi

  HIPAMD_CMAKE_OPTS=("${MYCMAKEOPTS[@]}"
                     -DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR;$HOME/local/openclicdloader;$BUILD_DIR/build/hipamd/opencl/khronos/icd"
                     -DCMAKE_INSTALL_LIBDIR=lib
                     -DCMAKE_CXX_FLAGS=-I"${AOMP_INSTALL_DIR}/include/amd_comgr"
                     -DCMAKE_CXX_FLAGS=-Wno-error=deprecated-declarations
                     -DCMAKE_C_FLAGS=-Wno-error=deprecated-declarations
                     -DHIP_LLVM_ROOT="$LLVM_INSTALL_LOC"
                     "${AOMP_ORIGIN_RPATH[@]}")

  echo "mkdir -p $BUILD_DIR/build/hipamd"
  mkdir -p "$BUILD_DIR/build/hipamd"
  echo "cd $BUILD_DIR/build/hipamd"
  cd "$BUILD_DIR/build/hipamd" || exit
  echo
  echo " -----Running hipamd cmake ---- "
  echo "${AOMP_CMAKE}" "${HIPAMD_CMAKE_OPTS[@]}" "$HIPAMD_DIR"

  if ! ${AOMP_CMAKE} "${HIPAMD_CMAKE_OPTS[@]}" "$HIPAMD_DIR"; then
      echo "ERROR hipamd cmake failed. Cmake flags"
      echo "      $(shquot "${HIPAMD_CMAKE_OPTS[@]}")"
      exit 1
  fi

  if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
     export ROCM_RPATH="$AOMP_ORIGIN_RPATH_LIST"
     echo "mkdir -p $BUILD_DIR/build/hipamd/asan"
     mkdir -p "$BUILD_DIR/build/hipamd/asan"
     echo "cd $BUILD_DIR/build/hipamd/asan"
     cd "$BUILD_DIR/build/hipamd/asan" || exit
     echo
     echo " -----Running hipamd-asan cmake -----"
     echo "${AOMP_CMAKE}" "${ASAN_CMAKE_OPTS[@]}" \
                          -DCMAKE_CXX_FLAGS=\""$(cmquot "${ASAN_FLAGS[@]}")\""
                          "$HIPAMD_DIR"
     
     if ! ${AOMP_CMAKE} "${ASAN_CMAKE_OPTS[@]}" \
                        -DCMAKE_CXX_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")" \
                        "$HIPAMD_DIR"; then
        echo "ERROR hipamd-asan cmake failed. Cmake flags"
        echo "      $(shquot "${ASAN_CMAKE_OPTS[@]}")"
        exit 1
     fi
  fi
  if [ "$AOMP_BUILD_DEBUG" == 1 ]; then
    if [ -d "$BUILD_DIR/build/hipamd_debug" ] ; then
       echo
       echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
       echo "rm -rf $BUILD_DIR/build/hipamd_debug"
       rm -rf "$BUILD_DIR/build/hipamd_debug"
    fi

     echo "mkdir -p $BUILD_DIR/build/hipamd_debug"
     mkdir -p "$BUILD_DIR/build/hipamd_debug"
     echo "cd $BUILD_DIR/build/hipamd_debug"
     cd "$BUILD_DIR/build/hipamd_debug" || exit
     echo
     echo " -----Running hipamd-debug cmake -----"
     _prefix_map=(-fdebug-prefix-map="$HIPAMD_DIR=$_ompd_src_dir/clr")
     echo "${AOMP_CMAKE}" "${HIPAMD_DEBUG_CMAKE_OPTS[@]}" \
          -DCMAKE_CXX_FLAGS="\"$(cmquot -g "${_prefix_map[@]}")\"" \
          -DCMAKE_C_FLAGS="\"$(cmquot -g "${_prefix_map[@]}")\"" \
          "$HIPAMD_DIR"

     if ! ${AOMP_CMAKE} "${HIPAMD_DEBUG_CMAKE_OPTS[@]}" \
             -DCMAKE_CXX_FLAGS="$(cmquot -g "${_prefix_map[@]}")" \
             -DCMAKE_C_FLAGS="$(cmquot -g "${_prefix_map[@]}")" \
             "$HIPAMD_DIR"; then
        echo "ERROR hipamd-debug cmake failed. Cmake flags"
        echo "      $(shquot "${HIPAMD_DEBUG_CMAKE_OPTS[@]}")"
        exit 1
     fi
  fi
fi

if [ "$1" = "cmake" ]; then
  exit 0
fi

cd "$BUILD_DIR/build/hipamd" || exit

echo
echo " -----Running make for hipamd ---- "

if ! make -j "$AOMP_JOB_THREADS"; then
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
   cd "$BUILD_DIR/build/hipamd/asan" || exit
   echo
   echo " -----Running make for hipamd-asan ----- "

   if ! make -j "$AOMP_JOB_THREADS"; then
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
if [ "$AOMP_BUILD_DEBUG" == 1 ] ; then
   cd "$BUILD_DIR/build/hipamd_debug" || exit
   echo
   echo " -----Running make for hipamd-debug ----- "

   if ! make -j "$AOMP_JOB_THREADS" amdhip64; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS FAILED"
      echo "To restart:"
      echo "  cd restart:"
      echo "  make "
      exit 1
   else
      if [ "$1" != "install" ] ; then
         echo
         echo " BUILD COMPLETE! To install hipamd-debug component run this command:"
         echo " $0 install"
         echo
      fi
   fi
fi

function edit_installed_hip_file(){
   local installed_file_to_edit="$1"
   if [ -f "$installed_file_to_edit" ] ; then
      # In hipvars.pm HIP_PATH is determined by parent directory of hipcc location.
      # Set ROCM_PATH using HIP_PATH
      $SUDO sed -i -e "s/\"\/opt\/rocm\"/\"\$HIP_PATH\"/" "$installed_file_to_edit"
      # Set HIP_CLANG_PATH using ROCM_PATH/bin
      $SUDO sed -i -e "s/\"\$ROCM_PATH\/llvm\/bin\"/\"\$ROCM_PATH\/bin\"/" "$installed_file_to_edit"
    fi
}

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
   cd "$BUILD_DIR/build/hipamd" || exit
   echo
   echo " -----Installing to $AOMP_INSTALL_DIR ----- "

   if ! $SUDO make install; then
      echo "ERROR make install failed "
      exit 1
   fi

   if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
      cd "$BUILD_DIR/build/hipamd/asan" || exit
      echo
      echo " -----Installing to $AOMP_INSTALL_DIR/lib/asan"

      if ! $SUDO make install; then
         echo "ERROR make install failed "
         exit 1
      fi
   fi
   if [ "$AOMP_BUILD_DEBUG" == 1 ] ; then
      cd "$BUILD_DIR/build/hipamd_debug" || exit
      echo
      echo " -----Installing to $AOMP_INSTALL_DIR/lib-debug"

      if ! $SUDO make install; then
         echo "ERROR make install failed "
         exit 1
      fi
      $SUDO mkdir -p "$_ompd_src_dir"
      echo  "cp -r $HIPAMD_DIR/hipamd $_ompd_src_dir"
      $SUDO cp -r "$HIPAMD_DIR/hipamd" "$_ompd_src_dir"
      echo  "cp -r $HIPAMD_DIR/opencl $_ompd_src_dir"
      $SUDO cp -r "$HIPAMD_DIR/opencl" "$_ompd_src_dir"
      echo  cp -r "$HIPAMD_DIR/rocclr" "$_ompd_src_dir"
      $SUDO cp -r "$HIPAMD_DIR/rocclr" "$_ompd_src_dir"
   fi

   removepatch "$AOMP_REPOS/hipamd"
   removepatch "$AOMP_REPOS/clr"

   # The hip perl scripts have /opt/rocm hardcoded, so fix them after then are installed
   # but only if not installing to rocm.
   if [ "$AOMP_INSTALL_DIR" != "/opt/rocm/llvm" ] ; then
      edit_installed_hip_file "$AOMP_INSTALL_DIR/bin/hipcc"
      edit_installed_hip_file "$AOMP_INSTALL_DIR/bin/hipvars.pm"
      # nothing to change in hipconfig but in case something is added in future, try to fix it
      edit_installed_hip_file "$AOMP_INSTALL_DIR/bin/hipconfig"
   fi
fi
