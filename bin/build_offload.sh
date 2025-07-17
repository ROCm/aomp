#!/bin/bash
#
#  build_offload.sh:  Script to build the AOMP runtime libraries and debug libraries.
#                This script will install in location defined by AOMP env variable
#

# Without these options, we can lose error status from command subtitutions,
# etc.
set -e
shopt -s inherit_errexit

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/aomp_utils"
. "$thisdir/aomp_common_vars"
# --- end standard header ----

# Get a configuration (environment) variable for this config
cfgvar() {
  get_config_var_string offload "$1"
}

cfgbool() {
  get_config_var_bool offload "$1"
}

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  help_build_aomp
fi

REPO_DIR=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME
_ompd_src_dir="$LLVM_INSTALL_LOC/share/gdb/python/ompd/src"

get_src_dir() {
   echo "$REPO_DIR/offload"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   local BuildDir
   BuildDir="$(cfgvar BUILD_DIR)"

   case "$Cfg" in
   "default")
     echo -n "$BuildDir/offload"
     ;;
   "asan")
     echo -n "$BuildDir/offload/asan"
     ;;
   "perf")
     echo -n "$BuildDir/offload_perf"
     ;;
   "perf+asan")
     echo -n "$BuildDir/offload_perf/asan"
     ;;
   "debug")
     echo -n "$BuildDir/offload_debug"
     ;;
   "debug+asan")
     echo -n "$BuildDir/offload_debug/asan"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   cfgvar LLVM_INSTALL_LOC
}

task_precheck() {
   local CUDAVER
   local SrcDir
   local CUDAH
   local CUDAVER
   local BuildCUDA
   local CUDATop
   local CUDAInclude
   local CUDABin
   local AOMPProc

   SrcDir="$(get_src_dir)"

   if [ ! -d "$SrcDir" ] ; then
      echo "ERROR:  Missing repository $SrcDir "
      echo "        Consider setting env variables AOMP_REPOS and/or AOMP_PROJECT_REPO_NAME "
      exit 1
   fi

   check_writable_installdir "$1" "$(cfgvar LLVM_INSTALL_LOC)"

   BuildCUDA="$(cfgbool AOMP_BUILD_CUDA)"

   if "$BuildCUDA"; then
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
         echo "        Please install CUDA to build offload"
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
}

task_clean() {
   local Cfg=$1
   local BuildDir
   BuildDir=$(get_build_dir "$1")
   echo "rm -rf $(shquot "$BuildDir")"
   rm -rf "$BuildDir"
}

task_patch() {
   local ApplyPatch
   local RepoDir
   ApplyPatch=$(cfgbool AOMP_APPLY_ATD_AMD_STAGING_PATCH)
   RepoDir=$(cfgvar REPO_DIR)
   # Patch llvm-project with ATD patch customized for amd-staging.
   # WARNING: This patch (ATD_ASO_full.patch) rarely applies cleanly
   #          because of its size and constant trunk merges to amd-staging.
   #          This is why default is 0 (OFF).
   if "$ApplyPatch" ; then
      patchrepo "$RepoDir"
   fi
}

task_unpatch() {
   local ApplyPatch
   local RepoDir
   ApplyPatch=$(cfgbool AOMP_APPLY_ATD_AMD_STAGING_PATCH)
   RepoDir=$(cfgvar REPO_DIR)
   if "$ApplyPatch" ; then
      removepatch "$RepoDir"
   fi
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

perf_config() {
   local Cfg=$1
   case "$Cfg" in
      perf|perf+*)
        return 0
        ;;
      *)
        ;;
   esac
   return 1
}

libdir_suffix() {
   local Cfg=$1
   case "$Cfg" in
     default)
       ;;
     asan)
       echo "/asan"
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
   esac
}

task_cmake() {
   local Cfg=$1
   local UseNinja
   local -a AOMP_SET_NINJA_GEN
   local GFXSEMICOLONS
   local ALTAOMP
   local Standalone
   local -a MYCMAKEOPTS
   local HSA_RUNTIME_PATH
   local BuildDir
   local _prefix_map
   local LibdirSuffix

   UseNinja=$(cfgbool AOMP_USE_NINJA)

   if "$UseNinja"; then
      AOMP_SET_NINJA_GEN=(-G Ninja)
   fi

   BuildDir=$(get_build_dir "$Cfg")

   export LLVM_DIR=$AOMP_INSTALL_DIR
   GFXSEMICOLONS=$(echo "$GFXLIST" | tr ' ' ';')
   ALTAOMP=${ALTAOMP:-$LLVM_INSTALL_LOC}

   Standalone=$(cfgbool AOMP_STANDALONE_BUILD)

   MYCMAKEOPTS=("${AOMP_SET_NINJA_GEN[@]}"
                -DOPENMP_ENABLE_LIBOMPTARGET=1
                -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_LOC"
                -DOPENMP_TEST_C_COMPILER="$LLVM_INSTALL_LOC/bin/clang"
                -DOPENMP_TEST_CXX_COMPILER="$LLVM_INSTALL_LOC/bin/clang++"
                -DCMAKE_C_COMPILER="$ALTAOMP/bin/clang"
                -DCMAKE_CXX_COMPILER="$ALTAOMP/bin/clang++"
                -DLIBOMPTARGET_AMDGCN_GFXLIST="$GFXSEMICOLONS"
                -DLLVM_DIR="$LLVM_DIR")

   if ! "$Standalone" ; then
      MYCMAKEOPTS+=(-DLLVM_MAIN_INCLUDE_DIR="$LLVM_PROJECT_ROOT/llvm/include"
                    -DLIBOMPTARGET_LLVM_INCLUDE_DIRS="$LLVM_PROJECT_ROOT/llvm/include")
   else
      MYCMAKEOPTS+=(-DLLVM_MAIN_INCLUDE_DIR="$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/llvm/include"
                    -DLIBOMPTARGET_LLVM_INCLUDE_DIRS="$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/llvm/include")
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

   #if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
      #LDFLAGS=$(shquot '-fuse-ld=lld' "${ASAN_FLAGS[@]}")
      #export LDFLAGS
   #fi

   # This is how we tell the hsa plugin where to find hsa
   export HSA_RUNTIME_PATH=$ROCM_DIR

   #breaks build as it cant find rocm-path
   #export HIP_DEVICE_LIB_PATH=$ROCM_DIR/lib

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit

   if perf_config "$Cfg"; then
      MYCMAKEOPTS+=(-DLIBOMPTARGET_PERF=ON
                    -DLIBOMPTARGET_ENABLE_DEBUG=OFF)
   else
      MYCMAKEOPTS+=(-DLIBOMPTARGET_ENABLE_DEBUG=ON)
   fi

   if debug_config "$Cfg"; then
      MYCMAKEOPTS+=(-DLIBOMPTARGET_NVPTX_DEBUG=ON
                    -DLLVM_ENABLE_ASSERTIONS=ON
                    -DROCM_DIR="$ROCM_DIR"
                    -DCMAKE_BUILD_TYPE=Debug
                    -DLIBOMP_ARCH=x86_64
                    -DLIBOMP_OMPT_SUPPORT=ON
                    -DLIBOMP_USE_DEBUGGER=ON
                    -DLIBOMP_CPPFLAGS='-O0'
                    -DLIBOMP_OMPD_SUPPORT=ON
                    -DLIBOMP_OMPT_DEBUG=ON)

      # The 'pip install --system' command is not supported on non-debian systems. This will disable
      # the system option if the debian_version file is not present.
      if [ ! -f /etc/debian_version ]; then
         echo "==> Non-Debian OS, disabling use of pip install --system"
         MYCMAKEOPTS+=(-DDISABLE_SYSTEM_NON_DEBIAN=1)
      fi

      # Redhat 7.6 does not have python36-devel package, which is needed for ompd compilation.
      # This is acquired through RH Software Collections.
      if [ -f /opt/rh/rh-python36/enable ]; then
         echo "==> Using python3.6 out of rh tools."
         MYCMAKEOPTS+=(-DPython3_ROOT_DIR=/opt/rh/rh-python36/root/bin
                       -DPYTHON_HEADERS=/opt/rh/rh-python36/root/usr/include/python3.6m)
      fi
   else
      MYCMAKEOPTS+=(-DCMAKE_BUILD_TYPE=Release)
   fi

   if asan_config "$Cfg"; then
      MYCMAKEOPTS+=(-DSANITIZER_AMDGPU=1
                    -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF)
   fi

   # rpath options
   if "$Standalone"; then
      if asan_config "$Cfg"; then
         MYCMAKEOPTS+=("${AOMP_ASAN_ORIGIN_RPATH[@]}"
                       -DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib/asan/cmake;$AOMP_INSTALL_DIR/lib/cmake;$AOMP_INSTALL_DIR/lib64/cmake")
      else
         if debug_config "$Cfg"; then
            MYCMAKEOPTS+=("${AOMP_DEBUG_ORIGIN_RPATH[@]}")
         else
            MYCMAKEOPTS+=("${AOMP_ORIGIN_RPATH[@]}")
         fi
         MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib/cmake;$AOMP_INSTALL_DIR/lib64/cmake")
      fi
   else
      if asan_config "$Cfg"; then
         MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$ROCM_CMAKECONFIG_PATH;$INSTALL_PREFIX/lib/llvm/lib/asan")
      else
         MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$INSTALL_PREFIX/lib/cmake")
      fi
      MYCMAKEOPTS+=("${OPENMP_EXTRAS_ORIGIN_RPATH[@]}")
   fi

   # libdir suffix options
   LibdirSuffix=$(libdir_suffix "$Cfg")

   if [ -n "$LibdirSuffix" ]; then
      MYCMAKEOPTS+=(-DOFFLOAD_LIBDIR_SUFFIX="$LibdirSuffix"
                    -DLLVM_LIBDIR_SUFFIX="$LibdirSuffix")
   fi

   # C/C++ flags
   if asan_config "$Cfg"; then
      MYCMAKEOPTS+=(-DCMAKE_C_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")"
                    -DCMAKE_CXX_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")")
   elif debug_config "$Cfg"; then
      _prefix_map=(-fdebug-prefix-map="$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload=$_ompd_src_dir/offload")
      MYCMAKEOPTS+=(-DCMAKE_C_FLAGS="$CFLAGS -g"
                    -DCMAKE_CXX_FLAGS="$CXXFLAGS -g $(cmquot "${_prefix_map[@]}")")
   fi

   echo "${AOMP_CMAKE}" "$(shquot "${MYCMAKEOPTS[@]}")" \
                        "$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload"
         
   if ! ${AOMP_CMAKE} "${MYCMAKEOPTS[@]}" \
                      "$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload"; then
      echo "ERROR offload cmake failed. Cmake flags"
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
   BuildDir=$(get_build_dir "$Cfg")
   NinjaBin="$(cfgvar AOMP_NINJA_BIN)"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"

   pushd "$BuildDir" >& /dev/null || exit
     echo " -----Running $NinjaBin for $BuildDir ---- "
      if ! $NinjaBin -j "$Jobs"; then
            echo " "
            echo "ERROR: $NinjaBin -j $Jobs FAILED"
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
   BuildDir=$(get_build_dir "$Cfg")
   NinjaBin="$(cfgvar AOMP_NINJA_BIN)"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"

   pushd "$BuildDir" >& /dev/null || exit

   echo " -----Installing to $LLVM_INSTALL_LOC/lib ----- "

   if ! $SUDO "$NinjaBin" -j "$Jobs" install; then
      echo "ERROR $NinjaBin install failed "
      exit 1
   fi

   popd >& /dev/null || exit
}

task_postinstall() {
   local Cfg=$1
   local Standalone
   local LLVMRoot
   local _from_dir_src
   local _from_dir_plugins
   
   Standalone="$(cfgbool AOMP_STANDALONE_BUILD)"
   LLVMRoot="$(cfgvar LLVM_PROJECT_ROOT)"

   # Copy selected debugable runtime sources into the installation directory
   # $_ompd_src_dir directory to satisfy the above CXXOPT  -fdebug-prefix-map.
   $SUDO mkdir -p "$_ompd_src_dir/offload"
   $SUDO mkdir -p "$_ompd_src_dir/offload/plugins-nextgen"
   if "$Standalone"; then
      _from_dir_src="$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload/libomptarget"
      _from_dir_plugins="$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload/plugins-nextgen"
   else
      _from_dir_src="$LLVMRoot/offload/libomptarget"
      _from_dir_plugins="$LLVMRoot/offload/plugins-nextgen"
   fi
   echo cp -rp "$_from_dir_src" "$_ompd_src_dir/offload"
   $SUDO cp -rp "$_from_dir_src" "$_ompd_src_dir/offload"
   echo cp -rp "$_from_dir_plugins" "$_ompd_src_dir/offload"
   $SUDO cp -rp "$_from_dir_plugins" "$_ompd_src_dir/offload"
}

do_list_configs() {
  local Sanitizer
  local LegacyOpenMP
  local BuildSanitizer
  local BuildPerf
  local BuildDebug

  Sanitizer="$(cfgbool SANITIZER)"
  LegacyOpenMP="$(cfgbool AOMP_LEGACY_OPENMP)"
  BuildSanitizer="$(cfgbool AOMP_BUILD_SANITIZER)"
  BuildPerf="$(cfgbool AOMP_BUILD_PERF)"
  BuildDebug="$(cfgbool AOMP_BUILD_DEBUG)"

  if ! "$Sanitizer" && "$LegacyOpenMP" ; then
    echo "default"
  fi
  if "$BuildSanitizer"; then
    echo "asan"
  fi
  if "$BuildPerf"; then
    echo "perf"
    if "$BuildSanitizer"; then
      echo "perf+asan"
    fi
  fi
  if "$BuildDebug" ; then
    if ! "$Sanitizer"; then
      echo "debug"
    fi
    if "$BuildSanitizer"; then
      echo "debug+asan"
    fi
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
