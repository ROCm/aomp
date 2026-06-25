#!/bin/bash
#
#  build_flang_runtime.sh:  Script to build the flang runtime component of the AOMP compiler.
#
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
  get_config_var_string flang_runtime "$1"
}

cfgbool() {
  get_config_var_bool flang_runtime "$1"
}

INSTALL_FLANG=${INSTALL_FLANG:-$AOMP_INSTALL_DIR}
REPO_DIR="$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_FLANG_REPO_NAME)"
COMP_INC_DIR="$REPO_DIR/runtime/libpgmath/lib/common"

if [ "$AOMP_PROC" == "ppc64le" ] ; then
   TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}PowerPC"
elif [ "$AOMP_PROC" == "aarch64" ] ; then
   TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}AArch64"
else
   TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}X86"
fi

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  help_build_aomp
fi

get_src_dir() {
   echo "$REPO_DIR"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   local BuildDir
   BuildDir="$(cfgvar BUILD_DIR)/flang_runtime"

   case "$Cfg" in
   "default")
     echo -n "$BuildDir"
     ;;
   "asan")
     echo -n "$BuildDir/asan"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   cfgvar INSTALL_FLANG
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

# Print the openmp runtime source directory used to drive OPENMP_BUILD_DIR,
# which depends on the config and on whether this is a standalone build.
get_openmp_runtime_src() {
   local Cfg=$1
   local BuildDir
   BuildDir="$(cfgvar BUILD_DIR)"

   if [ "$AOMP_STANDALONE_BUILD" == 0 ]; then
      if asan_config "$Cfg"; then
         echo -n "$BuildDir/openmp/asan/runtime/src"
      else
         echo -n "$BuildDir/openmp/runtime/src"
      fi
   else
      local Src
      if [ -d "$BuildDir/openmp/runtime/src" ] ; then
         Src="$BuildDir/openmp/runtime/src"
      else
         Src="$BuildDir/llvm-project/runtimes/runtimes-bins/openmp/runtime/src"
      fi
      if asan_config "$Cfg" && [ -d "$BuildDir/openmp/asan/runtime/src" ]; then
         Src="$BuildDir/openmp/asan/runtime/src"
      fi
      echo -n "$Src"
   fi
}

task_precheck() {
   local SrcDir
   SrcDir="$(get_src_dir)"

   if [ ! -d "$SrcDir" ] ; then
      echo "ERROR:  Missing repository $SrcDir"
      echo "        Are environment variables AOMP_REPOS and AOMP_FLANG_REPO_NAME set correctly?"
      exit 1
   fi

   check_writable_installdir "$1" "$(cfgvar INSTALL_FLANG)"
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
   local OpenmpRuntimeSrc
   local PgmathPrefix
   local -a MYCMAKEOPTS

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   OpenmpRuntimeSrc="$(get_openmp_runtime_src "$Cfg")"

   # Settings common to every config.
   MYCMAKEOPTS=(-DCMAKE_BUILD_TYPE="$(cfgvar BUILD_TYPE)"
                -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_LOC"
                -DLLVM_ENABLE_ASSERTIONS=ON
                -DLLVM_CONFIG="$LLVM_INSTALL_LOC/bin/llvm-config"
                -DCMAKE_CXX_COMPILER="$LLVM_INSTALL_LOC/bin/clang++"
                -DCMAKE_C_COMPILER="$LLVM_INSTALL_LOC/bin/clang"
                -DCMAKE_Fortran_COMPILER="$LLVM_INSTALL_LOC/bin/flang-classic"
                -DLLVM_TARGETS_TO_BUILD="$TARGETS_TO_BUILD"
                -DLLVM_INSTALL_RUNTIME=ON
                -DFLANG_BUILD_RUNTIME=ON
                -DOPENMP_BUILD_DIR="$OpenmpRuntimeSrc"
                -DFLANG_INCLUDE_TESTS=OFF)

   # Variant-specific settings.
   if asan_config "$Cfg"; then
      local -a _asan_flags
      _asan_flags=("${ASAN_FLAGS[@]}" -I"$COMP_INC_DIR")

      # Note this variable is used only for AOMP ASan-ROCm builds, don't remove
      # it.  This is not used for AOMP-ASan standalone build.
      if [ "$AOMP_STANDALONE_BUILD" == 0 ]; then
         MYCMAKEOPTS+=(-DSANITIZER=1)
      fi

      MYCMAKEOPTS+=(-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF
                    -DCMAKE_INSTALL_BINDIR=bin/asan
                    -DCMAKE_INSTALL_LIBDIR=lib/asan)
      if [ "$AOMP_STANDALONE_BUILD" == 1 ]; then
         MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$AOMP/lib/asan/cmake;$AOMP/lib/asan"
                       "${AOMP_ASAN_ORIGIN_RPATH[@]}")
      else
         PgmathPrefix="$(cfgvar BUILD_DIR)/pgmath/asan"
         MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$PgmathPrefix;$INSTALL_PREFIX/lib/asan/cmake"
                       "${OPENMP_EXTRAS_ORIGIN_RPATH[@]}"
                       -DOPENMP_EXTRAS_SHARED_LINKER_FLAGS="$(cmquot "${OPENMP_EXTRAS_SHARED_LINKER_FLAGS[@]}")")
      fi
      MYCMAKEOPTS+=(-DCMAKE_C_FLAGS="$CFLAGS $(cmquot "${_asan_flags[@]}")"
                    -DCMAKE_CXX_FLAGS="$CXXFLAGS $(cmquot "${_asan_flags[@]}")")
   else
      if [ "$AOMP_STANDALONE_BUILD" == 1 ]; then
         MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$AOMP/lib/cmake;$AOMP_INSTALL_DIR/lib"
                       "${AOMP_ORIGIN_RPATH[@]}")
      else
         PgmathPrefix="$(cfgvar BUILD_DIR)/pgmath"
         MYCMAKEOPTS+=(-DOPENMP_EXTRAS_SHARED_LINKER_FLAGS="$(cmquot "${OPENMP_EXTRAS_SHARED_LINKER_FLAGS[@]}")"
                       -DAOMP_BUILD_SANITIZER="$AOMP_BUILD_SANITIZER"
                       -DCMAKE_PREFIX_PATH="$PgmathPrefix"
                       "${OPENMP_EXTRAS_ORIGIN_RPATH[@]}")
      fi
      MYCMAKEOPTS+=(-DCMAKE_C_FLAGS="$CFLAGS -I$COMP_INC_DIR"
                    -DCMAKE_CXX_FLAGS="$CXXFLAGS -I$COMP_INC_DIR")
   fi

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running cmake for flang_runtime $Cfg ---- "
   echo "$AompCmake $(shquot "${MYCMAKEOPTS[@]}") $SrcDir"

   if ! "$AompCmake" "${MYCMAKEOPTS[@]}" "$SrcDir"; then
      echo "ERROR flang_runtime $Cfg cmake failed. Cmake flags"
      echo "      $(shquot "${MYCMAKEOPTS[@]}")"
      exit 1
   fi
   popd >& /dev/null || exit
}

task_build() {
   local Cfg=$1
   local BuildDir
   local Jobs
   BuildDir="$(get_build_dir "$Cfg")"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"

   #  Need llvm-config to come from previous LLVM build
   export PATH="$AOMP_INSTALL_DIR/bin":$PATH
   # Old CMAKE uses FC env variable to find fortran compiler
   export FC="$AOMP_INSTALL_DIR/bin/flang"

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running make for flang_runtime $Cfg ---- "
   echo "make -j $Jobs"
   if ! make -j "$Jobs"; then
      echo " "
      echo "ERROR: make -j $Jobs  FAILED"
      echo "To restart:"
      echo "  cd $BuildDir"
      echo "  make"
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

do_list_configs() {
  echo "default"
  if "$(cfgbool AOMP_BUILD_SANITIZER)"; then
    echo "asan"
  fi
}

do_list_init() {
  echo "precheck"
}

do_list_fini() {
  :
}

# List of tasks per config.
do_list_tasks() {
  local Cfg=$1
  if valid_config "$Cfg"; then
    echo "clean"
    echo "cmake"
    echo "build"
    echo "install"
  else
    echo "Unknown config '$Cfg'"
  fi
}

command_dispatcher "$@"
