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

# Without these options, we can lose error status from command subtitutions,
# etc.
set -e
shopt -s inherit_errexit

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath -- "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/aomp_utils"
. "$thisdir/aomp_common_vars"
# --- end standard header ----

# All user-controllable (environment) values are read through these wrappers so
# that they can later be driven by an orchestration layer.
cfgvar() {
  get_config_var_string hipamd "$1"
}

cfgbool() {
  get_config_var_bool hipamd "$1"
}

_repos="$(cfgvar AOMP_REPOS)"
export HIPAMD_DIR=$_repos/clr
export HIP_DIR=$_repos/hip
export ROCclr_DIR=$_repos/clr/rocclr
export OPENCL_DIR=$_repos/clr/opencl

export HSA_PATH=$AOMP_INSTALL_DIR
export ROCM_PATH=$AOMP_INSTALL_DIR
export HIP_CLANG_PATH=$AOMP_INSTALL_DIR/bin
export DEVICE_LIB_PATH=$AOMP_INSTALL_DIR/lib
export LLVM_DIR=$LLVM_INSTALL_LOC

_ompd_src_dir="$LLVM_INSTALL_LOC/share/gdb/python/ompd/src"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_hipamd.sh                   cmake, make, NO Install "
  echo "  ./build_hipamd.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_hipamd.sh install           NO Cmake, make install "
  echo " "
  exit 0
fi

get_src_dir() {
   echo "$HIPAMD_DIR"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   local BuildDir
   BuildDir="$(cfgvar BUILD_DIR)"

   case "$Cfg" in
   "default")
     echo -n "$BuildDir/hipamd"
     ;;
   "asan")
     echo -n "$BuildDir/hipamd/asan"
     ;;
   "debug")
     echo -n "$BuildDir/hipamd_debug"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   echo "$AOMP_INSTALL_DIR"
}

asan_config() {
   local Cfg=$1
   case "$Cfg" in
     asan|*+asan)
       return 0
       ;;
     *)
       ;;
   esac
   return 1
}

debug_config() {
   local Cfg=$1
   case "$Cfg" in
     debug|debug+*)
       return 0
       ;;
     *)
       ;;
   esac
   return 1
}

edit_installed_hip_file() {
   local installed_file_to_edit="$1"
   if [ -f "$installed_file_to_edit" ] ; then
      # In hipvars.pm HIP_PATH is determined by parent directory of hipcc location.
      # Set ROCM_PATH using HIP_PATH
      $SUDO sed -i -e "s/\"\/opt\/rocm\"/\"\$HIP_PATH\"/" "$installed_file_to_edit"
      # Set HIP_CLANG_PATH using ROCM_PATH/bin
      $SUDO sed -i -e "s/\"\$ROCM_PATH\/llvm\/bin\"/\"\$ROCM_PATH\/bin\"/" "$installed_file_to_edit"
   fi
}

task_precheck() {
   [[ ! -d $HIPAMD_DIR ]] && echo "ERROR:  Missing $HIPAMD_DIR" && exit 1
   [[ ! -d $HIP_DIR ]]    && echo "ERROR:  Missing $HIP_DIR"    && exit 1
   [[ ! -d $ROCclr_DIR ]] && echo "ERROR:  Missing $ROCclr_DIR" && exit 1
   [[ ! -d $OPENCL_DIR ]] && echo "ERROR:  Missing $OPENCL_DIR" && exit 1

   check_writable_installdir "$1" "$AOMP_INSTALL_DIR"
   return 0
}

task_patch() {
   patchrepo "$_repos/hipamd"
   patchrepo "$_repos/clr"
}

task_unpatch() {
   removepatch "$_repos/hipamd"
   removepatch "$_repos/clr"
}

task_clean() {
   local Cfg=$1
   local BuildDir
   BuildDir=$(get_build_dir "$Cfg")
   echo "rm -rf $(shquot "$BuildDir")"
   rm -rf "$BuildDir"
}

task_cmake() {
   local Cfg=$1
   local BuildDir
   local SrcDir
   local AompCmake
   local BuildRoot
   local Gfx
   local amdgpu
   local -a MYCMAKEOPTS
   local -a CMAKEOPTS
   local -a _flags
   local -a _prefix_map

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   BuildRoot="$(cfgvar BUILD_DIR)"
   Gfx="$(cfgvar GFXLIST)"

   # Settings common to every config.
   MYCMAKEOPTS=(-DCMAKE_BUILD_TYPE=Release
                -DCMAKE_INSTALL_PREFIX="$AOMP_INSTALL_DIR"
                -DHIP_COMMON_DIR="$HIP_DIR"
                -DHIP_PLATFORM=amd
                -DHIP_COMPILER=clang
                -DCMAKE_HIP_ARCHITECTURES=OFF
                -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=ON
                -DHIPCC_BIN_DIR="$BuildRoot/hipcc"
                -DROCM_PATH="$ROCM_PATH"
                -DBUILD_ICD=ON)

   # If this machine does not have an active amd GPU, tell hipamd
   # to use first in GFXLIST or gfx90a if no GFXLIST
   if [ -f "$LLVM_INSTALL_LOC/bin/amdgpu-arch" ] ; then
      if ! "$LLVM_INSTALL_LOC/bin/amdgpu-arch" >/dev/null; then
         if [ -n "$Gfx" ] ; then
            amdgpu=$(echo "$Gfx" | cut -d" " -f1)
         else
            amdgpu=gfx90a
         fi
         MYCMAKEOPTS+=("-DOFFLOAD_ARCH_STR=$amdgpu")
      fi
   fi

   # Variant-specific settings.
   if asan_config "$Cfg"; then
      _flags=("${ASAN_FLAGS[@]}" -I"$SANITIZER_COMGR_INCLUDE_PATH" -Wno-error=deprecated-declarations)
      export ROCM_RPATH="$AOMP_ORIGIN_RPATH_LIST"
      CMAKEOPTS=("${MYCMAKEOPTS[@]}"
                 "${AOMP_ASAN_ORIGIN_RPATH[@]}"
                 -DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib/asan/cmake;$AOMP_INSTALL_DIR;$HOME/local/openclicdloader;$BuildRoot/hipamd/opencl/khronos/icd"
                 -DCMAKE_INSTALL_LIBDIR=lib/asan
                 -DCMAKE_C_COMPILER="$LLVM_INSTALL_LOC/bin/clang"
                 -DCMAKE_CXX_COMPILER="$LLVM_INSTALL_LOC/bin/clang++"
                 -DHIP_LLVM_ROOT="$LLVM_INSTALL_LOC"
                 -DCMAKE_CXX_FLAGS="$(cmquot "${_flags[@]}")")
   elif debug_config "$Cfg"; then
      _prefix_map=(-fdebug-prefix-map="$HIPAMD_DIR=$_ompd_src_dir/clr")
      CMAKEOPTS=("${MYCMAKEOPTS[@]}"
                 "${AOMP_DEBUG_ORIGIN_RPATH[@]}"
                 -DCMAKE_BUILD_TYPE=DEBUG
                 -DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR;$HOME/local/openclicdloader;$BuildRoot/hipamd/opencl/khronos/icd"
                 -DCMAKE_INSTALL_LIBDIR=lib-debug
                 -DCMAKE_C_COMPILER="$LLVM_INSTALL_LOC/bin/clang"
                 -DCMAKE_CXX_COMPILER="$LLVM_INSTALL_LOC/bin/clang++"
                 -DHIP_LLVM_ROOT="$LLVM_INSTALL_LOC"
                 -DCMAKE_CXX_FLAGS="$(cmquot -g "${_prefix_map[@]}")"
                 -DCMAKE_C_FLAGS="$(cmquot -g "${_prefix_map[@]}")")
   else
      CMAKEOPTS=("${MYCMAKEOPTS[@]}"
                 -DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR;$HOME/local/openclicdloader;$BuildRoot/hipamd/opencl/khronos/icd"
                 -DCMAKE_INSTALL_LIBDIR=lib
                 -DCMAKE_CXX_FLAGS=-I"${AOMP_INSTALL_DIR}/include/amd_comgr"
                 -DCMAKE_CXX_FLAGS=-Wno-error=deprecated-declarations
                 -DCMAKE_C_FLAGS=-Wno-error=deprecated-declarations
                 -DHIP_LLVM_ROOT="$LLVM_INSTALL_LOC"
                 "${AOMP_ORIGIN_RPATH[@]}")
   fi

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running hipamd $Cfg cmake ---- "
   echo "$AompCmake $(shquot "${CMAKEOPTS[@]}") $SrcDir"

   if ! "$AompCmake" "${CMAKEOPTS[@]}" "$SrcDir"; then
      echo "ERROR hipamd $Cfg cmake failed. Cmake flags"
      echo "      $(shquot "${CMAKEOPTS[@]}")"
      exit 1
   fi
   popd >& /dev/null || exit
}

task_build() {
   local Cfg=$1
   local BuildDir
   local Jobs
   local -a MakeArgs
   BuildDir="$(get_build_dir "$Cfg")"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"

   # The debug config only builds the amdhip64 target.
   MakeArgs=(-j "$Jobs")
   if debug_config "$Cfg"; then
      MakeArgs+=(amdhip64)
   fi

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running make for hipamd $Cfg ---- "
   echo "make ${MakeArgs[*]}"
   if ! make "${MakeArgs[@]}"; then
      echo " "
      echo "ERROR: make -j $Jobs  FAILED"
      echo "To restart:"
      echo "  cd $BuildDir"
      echo "  make "
      exit 1
   fi
   popd >& /dev/null || exit
}

task_install() {
   local Cfg=$1
   local BuildDir
   local InstallDir
   BuildDir="$(get_build_dir "$Cfg")"
   InstallDir="$(get_install_dir "$Cfg")"

   pushd "$BuildDir" >& /dev/null || exit
   if asan_config "$Cfg"; then
      echo " -----Installing to $InstallDir/lib/asan ----- "
   elif debug_config "$Cfg"; then
      echo " -----Installing to $InstallDir/lib-debug ----- "
   else
      echo " -----Installing to $InstallDir ----- "
   fi
   echo "$SUDO make install "

   if ! $SUDO make install; then
      echo "ERROR make install failed "
      exit 1
   fi
   popd >& /dev/null || exit
}

task_postinstall() {
   local Cfg=$1

   if debug_config "$Cfg"; then
      # copy hipamd sources into the installation for runtime source debugging
      $SUDO mkdir -p "$_ompd_src_dir"
      echo "cp -r $HIPAMD_DIR/hipamd $_ompd_src_dir"
      $SUDO cp -r "$HIPAMD_DIR/hipamd" "$_ompd_src_dir"
      echo "cp -r $HIPAMD_DIR/opencl $_ompd_src_dir"
      $SUDO cp -r "$HIPAMD_DIR/opencl" "$_ompd_src_dir"
      echo "cp -r $HIPAMD_DIR/rocclr $_ompd_src_dir"
      $SUDO cp -r "$HIPAMD_DIR/rocclr" "$_ompd_src_dir"
      return 0
   fi

   # The hip perl scripts have /opt/rocm hardcoded, so fix them after they are
   # installed but only if not installing to rocm.
   if [ "$AOMP_INSTALL_DIR" != "/opt/rocm/llvm" ] ; then
      edit_installed_hip_file "$AOMP_INSTALL_DIR/bin/hipcc"
      edit_installed_hip_file "$AOMP_INSTALL_DIR/bin/hipvars.pm"
      # nothing to change in hipconfig but in case something is added in future, try to fix it
      edit_installed_hip_file "$AOMP_INSTALL_DIR/bin/hipconfig"
   fi
}

do_list_configs() {
  echo "default"
  if "$(cfgbool AOMP_BUILD_SANITIZER)"; then
    echo "asan"
  fi
  if "$(cfgbool AOMP_BUILD_DEBUG)"; then
    echo "debug"
  fi
}

do_list_init() {
  echo "precheck"
  echo "patch"
}

do_list_fini() {
  echo "unpatch"
}

# List of tasks per config.
do_list_tasks() {
  local Cfg=$1
  if valid_config "$Cfg"; then
    echo "clean"
    echo "cmake"
    echo "build"
    echo "install"
    case "$Cfg" in
      default|debug)
        echo "postinstall"
        ;;
      *)
        ;;
    esac
  else
    echo "Unknown config '$Cfg'"
  fi
}

command_dispatcher "$@"
