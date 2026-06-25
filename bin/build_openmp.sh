#!/bin/bash
#
#  build_openmp.sh:  Script to build the AOMP runtime libraries and debug libraries.
#                This script will install in location defined by AOMP env variable
#

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
  get_config_var_string openmp "$1"
}

cfgbool() {
  get_config_var_bool openmp "$1"
}

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  help_build_aomp
fi

REPO_DIR=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME
_ompd_src_dir="$LLVM_INSTALL_LOC/share/gdb/python/ompd/src"
OPENMP_BUILD_DIR=${OPENMP_BUILD_DIR:-"openmp"}
ALTAOMP=${ALTAOMP:-$LLVM_INSTALL_LOC}

get_src_dir() {
   echo "$REPO_DIR/openmp"
}

# Return success if the config is a "device runtime library" pass, which builds
# the same set of variants in a separate build directory with the amdgcn
# default target triple.
devicertl_config() {
   local Cfg=$1
   case "$Cfg" in
     *-devicertl)
       return 0
       ;;
     *)
       ;;
   esac
   return 1
}

asan_config() {
   local Cfg=${1%-devicertl}
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
   local Cfg=${1%-devicertl}
   case "$Cfg" in
     debug|debug+*)
       return 0
       ;;
     *)
       ;;
   esac
   return 1
}

perf_config() {
   local Cfg=${1%-devicertl}
   case "$Cfg" in
     perf|perf+*)
       return 0
       ;;
     *)
       ;;
   esac
   return 1
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   local Base=${Cfg%-devicertl}
   local BuildDir
   local DirBase
   BuildDir="$(cfgvar BUILD_DIR)"

   if devicertl_config "$Cfg"; then
      DirBase="openmp-devicertl"
   else
      DirBase="$(cfgvar OPENMP_BUILD_DIR)"
   fi

   case "$Base" in
   "default")
     echo -n "$BuildDir/$DirBase"
     ;;
   "asan")
     echo -n "$BuildDir/$DirBase/asan"
     ;;
   "perf")
     echo -n "$BuildDir/${DirBase}_perf"
     ;;
   "perf+asan")
     echo -n "$BuildDir/${DirBase}_perf/asan"
     ;;
   "debug")
     echo -n "$BuildDir/${DirBase}_debug"
     ;;
   "debug+asan")
     echo -n "$BuildDir/${DirBase}_debug/asan"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   echo "$LLVM_INSTALL_LOC"
}

# Print the LLVM_LIBDIR_SUFFIX for a given config.
libdir_suffix() {
   local Cfg=${1%-devicertl}
   case "$Cfg" in
     default)
       ;;
     asan)
       printf "%s" "/asan"
       ;;
     perf)
       printf "%s" "-perf"
       ;;
     perf+asan)
       printf "%s" "-perf/asan"
       ;;
     debug)
       printf "%s" "-debug"
       ;;
     debug+asan)
       printf "%s" "-debug/asan"
       ;;
     *)
       >&2 echo "Unknown config '$Cfg'"
       exit 1
       ;;
   esac
}

task_precheck() {
   local SrcDir
   local CUDAH
   local CUDAVER
   local CUDATop
   local CUDAInclude
   local CUDABin
   local AOMPProc

   SrcDir="$(get_src_dir)"

   if [ ! -d "$REPO_DIR" ] ; then
      echo "ERROR:  Missing repository $REPO_DIR "
      echo "        Consider setting env variables AOMP_REPOS and/or AOMP_PROJECT_REPO_NAME "
      exit 1
   fi

   check_writable_installdir "$1" "$LLVM_INSTALL_LOC"

   if "$(cfgbool AOMP_BUILD_CUDA)"; then
      CUDATop=$(cfgvar CUDAT)
      CUDAH=$(find "$CUDATop" -type f,l -name "cuda.h" 2>/dev/null)
      if [ "$CUDAH" == "" ] ; then
         CUDAInclude=$(cfgvar CUDAINCLUDE)
         CUDAH=$(find "$CUDAInclude" -type f,l -name "cuda.h" 2>/dev/null)
      fi
      if [ "$CUDAH" == "" ] ; then
         AOMPProc=$(cfgvar AOMP_PROC)
         echo
         echo "ERROR:  THE cuda.h FILE WAS NOT FOUND WITH ARCH $AOMPProc"
         echo "        A CUDA installation is necessary to build libomptarget deviceRTLs"
         echo "        Please install CUDA to build openmp"
         echo
         exit 1
      fi
      # I don't see now nvcc is called, but this eliminates the deprecated warnings
      export CUDAFE_FLAGS="-w"

      CUDABin=$(cfgvar CUDABIN)
      if [ -f "$CUDABin/nvcc" ] ; then
         CUDAVER=$("$CUDABin"/nvcc --version | grep compilation | cut -d" " -f5 | cut -d"." -f1)
         echo "CUDA VERSION IS $CUDAVER"
      fi
   fi
   return 0
}

task_patch() {
   # Patch llvm-project with ATD patch customized for amd-staging.
   # WARNING: This patch (ATD_ASO_full.patch) rarely applies cleanly
   #          because of its size and constant trunk merges to amd-staging.
   #          This is why default is 0 (OFF).
   if "$(cfgbool AOMP_APPLY_ATD_AMD_STAGING_PATCH)" ; then
      patchrepo "$REPO_DIR"
   fi
}

task_unpatch() {
   if "$(cfgbool AOMP_APPLY_ATD_AMD_STAGING_PATCH)" ; then
      removepatch "$REPO_DIR"
   fi
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
   local SrcDir
   local BuildDir
   local AompCmake
   local UseNinja
   local Standalone
   local GFXSEMICOLONS
   local LibdirSuffix
   local -a AOMP_SET_NINJA_GEN
   local -a MYCMAKEOPTS
   local -a DEBUGCMAKEOPTS

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   UseNinja="$(cfgbool AOMP_USE_NINJA)"
   Standalone="$(cfgbool AOMP_STANDALONE_BUILD)"

   if "$UseNinja"; then
      AOMP_SET_NINJA_GEN=(-G Ninja)
   fi

   export LLVM_DIR=$AOMP_INSTALL_DIR
   GFXSEMICOLONS=$(echo "$GFXLIST" | tr ' ' ';')

   # This is how we tell the hsa plugin where to find hsa
   export HSA_RUNTIME_PATH=$ROCM_DIR

   # Settings common to every config.
   MYCMAKEOPTS=("${AOMP_SET_NINJA_GEN[@]}"
                -DOPENMP_ENABLE_LIBOMPTARGET=1
                -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_LOC"
                -DOPENMP_TEST_C_COMPILER="$LLVM_INSTALL_LOC/bin/clang"
                -DOPENMP_TEST_CXX_COMPILER="$LLVM_INSTALL_LOC/bin/clang++"
                -DCMAKE_C_COMPILER="$ALTAOMP/bin/clang"
                -DCMAKE_CXX_COMPILER="$ALTAOMP/bin/clang++"
                -DLIBOMPTARGET_AMDGCN_GFXLIST="$GFXSEMICOLONS"
                -DDEVICELIBS_ROOT="$DEVICELIBS_ROOT"
                -DLIBOMP_COPY_EXPORTS=OFF
                -DLIBOMPTARGET_ENABLE_DEBUG=ON
                -DLIBOMPTEST_INSTALL_COMPONENTS=ON
                -DLLVM_DIR="$LLVM_DIR")

   # The device runtime library pass builds for the amdgcn default triple.
   if devicertl_config "$Cfg"; then
      if [ -f "$REPO_DIR/openmp/device/CMakeLists.txt" ]; then
         MYCMAKEOPTS+=(-DLLVM_DEFAULT_TARGET_TRIPLE=amdgcn-amd-amdhsa)
      fi
   fi

   if ! "$Standalone"; then
      # For static package builds, set BUILD_SHARED_LIBS to OFF
      if [ "$STATIC_PKG_DEPS" == "ON" ]; then
         MYCMAKEOPTS+=(-DBUILD_SHARED_LIBS=OFF)
      fi
      MYCMAKEOPTS+=(-DLLVM_MAIN_INCLUDE_DIR="$LLVM_PROJECT_ROOT/llvm/include"
                    -DLIBOMPTARGET_LLVM_INCLUDE_DIRS="$LLVM_PROJECT_ROOT/llvm/include"
                    -DROCM_DIR="$ROCM_DIR"
                    -DAOMP_STANDALONE_BUILD="$(cfgvar AOMP_STANDALONE_BUILD)")
   else
      MYCMAKEOPTS+=(-DLLVM_MAIN_INCLUDE_DIR="$REPO_DIR/llvm/include"
                    -DLIBOMPTARGET_LLVM_INCLUDE_DIRS="$REPO_DIR/llvm/include"
                    -DLIBOMP_USE_HWLOC=ON
                    -DLIBOMP_HWLOC_INSTALL_DIR="$AOMP_SUPP/hwloc")
   fi

   if "$(cfgbool AOMP_BUILD_CUDA)"; then
      MYCMAKEOPTS+=(-DLIBOMPTARGET_NVPTX_ENABLE_BCLIB=ON
                    -DLIBOMPTARGET_NVPTX_CUDA_COMPILER="$AOMP/bin/clang++"
                    -DLIBOMPTARGET_NVPTX_BC_LINKER="$AOMP/bin/llvm-link"
                    -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES="$NVPTXGPUS")
   else
      #  Need to force CUDA off this way in case cuda is installed in this system
      MYCMAKEOPTS+=(-DCUDA_TOOLKIT_ROOT_DIR=OFF)
   fi

   # Variant-specific settings.
   if perf_config "$Cfg"; then
      MYCMAKEOPTS+=(-DLIBOMPTARGET_ENABLE_DEBUG=OFF
                    -DCMAKE_BUILD_TYPE=Release
                    -DLIBOMPTARGET_PERF=ON)
      if asan_config "$Cfg"; then
         MYCMAKEOPTS+=(-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF
                       -DSANITIZER_AMDGPU=1
                       -DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib/asan/cmake;$AOMP_INSTALL_DIR/lib/cmake/AMDDeviceLibs"
                       "${AOMP_ASAN_ORIGIN_RPATH[@]}"
                       -DCMAKE_C_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")"
                       -DCMAKE_CXX_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")")
      else
         MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib/cmake"
                       "${AOMP_ORIGIN_RPATH[@]}")
      fi
   elif debug_config "$Cfg"; then
      DEBUGCMAKEOPTS=(-DLIBOMPTARGET_NVPTX_DEBUG=ON
                      -DLLVM_ENABLE_ASSERTIONS=ON
                      -DCMAKE_BUILD_TYPE=Debug
                      -DROCM_DIR="$ROCM_DIR"
                      -DLIBOMP_ARCH=x86_64
                      -DLIBOMP_OMPT_SUPPORT=ON
                      -DLIBOMP_USE_DEBUGGER=ON
                      -DLIBOMP_CPPFLAGS='-O0'
                      -DLIBOMP_OMPD_SUPPORT=ON
                      -DLIBOMP_OMPT_DEBUG=ON
                      -DOPENMP_SOURCE_DEBUG_MAP="-fdebug-prefix-map=$REPO_DIR/openmp=$_ompd_src_dir/openmp")

      # The 'pip install --system' command is not supported on non-debian systems. This will disable
      # the system option if the debian_version file is not present.
      if [ ! -f /etc/debian_version ]; then
         echo "==> Non-Debian OS, disabling use of pip install --system"
         DEBUGCMAKEOPTS+=(-DDISABLE_SYSTEM_NON_DEBIAN=1)
      fi

      # Redhat 7.6 does not have python36-devel package, which is needed for ompd compilation.
      # This is acquired through RH Software Collections.
      if [ -f /opt/rh/rh-python36/enable ]; then
         echo "==> Using python3.6 out of rh tools."
         DEBUGCMAKEOPTS+=(-DPython3_ROOT_DIR=/opt/rh/rh-python36/root/bin
                          -DPYTHON_HEADERS=/opt/rh/rh-python36/root/usr/include/python3.6m)
      fi

      MYCMAKEOPTS+=("${DEBUGCMAKEOPTS[@]}")

      if asan_config "$Cfg"; then
         MYCMAKEOPTS+=(-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF
                       -DSANITIZER_AMDGPU=1)
         if "$Standalone"; then
            MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib/asan/cmake;$AOMP_INSTALL_DIR/lib/cmake/AMDDeviceLibs"
                          "${AOMP_ASAN_ORIGIN_RPATH[@]}")
         else
            MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$ROCM_CMAKECONFIG_PATH"
                          "${OPENMP_EXTRAS_ORIGIN_RPATH[@]}")
         fi
         MYCMAKEOPTS+=(-DCMAKE_C_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")"
                       -DCMAKE_CXX_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")")
      else
         if "$Standalone"; then
            MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib/cmake"
                          "${AOMP_DEBUG_ORIGIN_RPATH[@]}")
         else
            MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$INSTALL_PREFIX/lib/cmake"
                          "${OPENMP_EXTRAS_ORIGIN_RPATH[@]}")
         fi
         MYCMAKEOPTS+=(-DCMAKE_C_FLAGS="$CFLAGS -g"
                       -DCMAKE_CXX_FLAGS="$CXXFLAGS -g")
      fi
   elif asan_config "$Cfg"; then
      MYCMAKEOPTS+=(-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF
                    -DSANITIZER_AMDGPU=1
                    -DCMAKE_BUILD_TYPE=Release)
      if "$Standalone"; then
         MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib/asan/cmake;$AOMP_INSTALL_DIR/lib/cmake/AMDDeviceLibs"
                       "${AOMP_ASAN_ORIGIN_RPATH[@]}")
      else
         MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$ROCM_CMAKECONFIG_PATH"
                       "${OPENMP_EXTRAS_ORIGIN_RPATH[@]}")
      fi
      MYCMAKEOPTS+=(-DCMAKE_C_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")"
                    -DCMAKE_CXX_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")")
   else
      MYCMAKEOPTS+=(-DCMAKE_BUILD_TYPE=Release)
      if "$Standalone"; then
         MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib/cmake"
                       "${AOMP_ORIGIN_RPATH[@]}")
      else
         MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$INSTALL_PREFIX/lib/cmake"
                       "${OPENMP_EXTRAS_ORIGIN_RPATH[@]}")
         # XXX: Crude way to detect if we should enable building the mod files with flang.
         if [ -e "${LLVM_INSTALL_LOC}/bin/flang" ]; then
            echo "Building .mod files via:  $LLVM_INSTALL_LOC/bin/flang"
            echo "Installing .mod files to: $LLVM_INSTALL_LOC/include/flang/"
            MYCMAKEOPTS+=(-DLIBOMP_FORTRAN_MODULES_COMPILER="$LLVM_INSTALL_LOC/bin/flang"
                          -DLIBOMP_MODULES_INSTALL_PATH="$LLVM_INSTALL_LOC/include/flang/")
         fi
      fi
   fi

   # libdir suffix options
   LibdirSuffix=$(libdir_suffix "$Cfg")
   if [ -n "$LibdirSuffix" ]; then
      MYCMAKEOPTS+=(-DLLVM_LIBDIR_SUFFIX="$LibdirSuffix")
   fi

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running openmp $Cfg cmake ---- "
   echo "$AompCmake $(shquot "${MYCMAKEOPTS[@]}") $SrcDir"

   if ! "$AompCmake" "${MYCMAKEOPTS[@]}" "$SrcDir"; then
      echo "ERROR openmp $Cfg cmake failed. Cmake flags"
      echo "      $(shquot "${MYCMAKEOPTS[@]}")"
      exit 1
   fi
   popd >& /dev/null || exit
}

task_build() {
   local Cfg=$1
   local BuildDir
   local NinjaBin
   local Jobs
   BuildDir="$(get_build_dir "$Cfg")"
   NinjaBin="$(cfgvar AOMP_NINJA_BIN)"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running $NinjaBin for $BuildDir ---- "
   if ! $NinjaBin -j "$Jobs"; then
      echo " "
      echo "ERROR: $NinjaBin -j $Jobs  FAILED"
      echo "To restart:"
      echo "  cd $BuildDir"
      echo "  $NinjaBin"
      exit 1
   fi
   popd >& /dev/null || exit
}

task_install() {
   local Cfg=$1
   local BuildDir
   local NinjaBin
   local Jobs
   local LibdirSuffix
   BuildDir="$(get_build_dir "$Cfg")"
   NinjaBin="$(cfgvar AOMP_NINJA_BIN)"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"
   LibdirSuffix=$(libdir_suffix "$Cfg")

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Installing to $LLVM_INSTALL_LOC/lib${LibdirSuffix} ----- "

   if ! $SUDO "$NinjaBin" -j "$Jobs" install; then
      echo "ERROR $NinjaBin install failed "
      exit 1
   fi
   popd >& /dev/null || exit
}

task_postinstall() {
   local Cfg=$1
   local Standalone
   Standalone="$(cfgbool AOMP_STANDALONE_BUILD)"

   # Copy selected debugable runtime sources into the installation
   # $_ompd_src_dir directory to satisfy the above -fdebug-prefix-map.
   $SUDO mkdir -p "$_ompd_src_dir/openmp/runtime"
   $SUDO mkdir -p "$_ompd_src_dir/openmp/libompd"
   $SUDO mkdir -p "$_ompd_src_dir/openmp/device"
   if "$Standalone"; then
      $SUDO cp -rp "$REPO_DIR/openmp/runtime/src" "$_ompd_src_dir/openmp/runtime"
      $SUDO cp -rp "$REPO_DIR/openmp/libompd/src" "$_ompd_src_dir/openmp/libompd"
      $SUDO cp -rp "$REPO_DIR/openmp/device/src" "$_ompd_src_dir/openmp/device"
   else
      $SUDO cp -rp "$LLVM_PROJECT_ROOT/openmp/runtime/src" "$_ompd_src_dir/openmp/runtime"
      $SUDO cp -rp "$LLVM_PROJECT_ROOT/openmp/libompd/src" "$_ompd_src_dir/openmp/libompd"
      $SUDO cp -rp "$LLVM_PROJECT_ROOT/openmp/device/src" "$_ompd_src_dir/openmp/device"
   fi
}

do_list_configs() {
  local Sanitizer
  local LegacyOpenMP
  local BuildSanitizer
  local BuildPerf
  local BuildDebug
  local -a cfgs
  local c

  Sanitizer="$(cfgbool SANITIZER)"
  LegacyOpenMP="$(cfgbool AOMP_LEGACY_OPENMP)"
  BuildSanitizer="$(cfgbool AOMP_BUILD_SANITIZER)"
  BuildPerf="$(cfgbool AOMP_BUILD_PERF)"
  BuildDebug="$(cfgbool AOMP_BUILD_DEBUG)"

  cfgs=()
  if ! "$Sanitizer" && "$LegacyOpenMP" ; then
    cfgs+=("default")
  fi
  if "$BuildSanitizer"; then
    cfgs+=("asan")
  fi
  if "$BuildPerf"; then
    cfgs+=("perf")
    if "$BuildSanitizer"; then
      cfgs+=("perf+asan")
    fi
  fi
  if "$BuildDebug" ; then
    if ! "$Sanitizer"; then
      cfgs+=("debug")
    fi
    if "$BuildSanitizer"; then
      cfgs+=("debug+asan")
    fi
  fi

  # First build/install every variant in the host build dir, then repeat the
  # whole set as a device runtime library pass in the openmp-devicertl dir.
  for c in "${cfgs[@]}"; do
    echo "$c"
  done
  for c in "${cfgs[@]}"; do
    echo "$c-devicertl"
  done
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
      debug)
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
