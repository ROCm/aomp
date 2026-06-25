#!/bin/bash
#
#  build_rocgdb.sh:  Script to build ROCgdb for aomp standalone build
#                 This will be called by build_aomp.sh when
#                 AOMP_STANDALONE_BUILD=1 && AOMP_BUILD_DEBUG==1
#                 This depends on rocdbgapi to be built and installed.
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
  get_config_var_string rocgdb "$1"
}

cfgbool() {
  get_config_var_bool rocgdb "$1"
}

# Point to the right python3.6 on Red Hat 7.6
if [ -f /opt/rh/rh-python36/enable ]; then
  export PATH=/opt/rh/rh-python36/root/usr/bin:$PATH
  export LIBRARY_PATH=/opt/rh/rh-python36/root/lib64:$LIBRARY_PATH
fi

REPO_DIR="$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_GDB_REPO_NAME)"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo " This script builds ROCgdb for AOMP standalone build"
  echo " It gets the source from:  $REPO_DIR"
  echo " It builds libraries in:   $(cfgvar BUILD_DIR)/rocgdb"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocgdb.sh                   configure, make , NO Install "
  echo "  ./build_rocgdb.sh noconfigure       NO configure, make, NO install "
  echo "  ./build_rocgdb.sh install           NO configure, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 0
fi

get_src_dir() {
   echo "$REPO_DIR"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   local BuildDir
   BuildDir="$(cfgvar BUILD_DIR)"

   case "$Cfg" in
   "default")
     echo -n "$BuildDir/rocgdb"
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

task_precheck() {
   local SrcDir
   SrcDir="$(get_src_dir)"

   if [ ! -d "$SrcDir" ] ; then
      echo "ERROR:  Missing repository $SrcDir"
      echo "        Are environment variables AOMP_REPOS and AOMP_GDB_REPO_NAME set correctly?"
      exit 1
   fi

   check_writable_installdir "$1" "$AOMP_INSTALL_DIR"
}

task_clean() {
   local Cfg=$1
   local BuildDir
   BuildDir=$(get_build_dir "$Cfg")
   echo "rm -rf $(shquot "$BuildDir")"
   rm -rf "$BuildDir"
}

# ROCgdb uses an autotools configure step rather than cmake; it is named
# task_cmake so that it runs in the standard "cmake" phase of the dispatcher.
task_cmake() {
   local Cfg=$1
   local BuildDir
   local SrcDir
   local BugUrl
   local -a MYCONFIGOPTS

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   BugUrl="https://github.com/ROCm/ROCgdb/issues"

   export CXXFLAGS_FOR_BUILD="-O2"
   export CFLAGS_FOR_BUILD="-O2"

   MYCONFIGOPTS=(--prefix="$AOMP_INSTALL_DIR"
                 --srcdir="$SrcDir"
                 --program-prefix=roc
                 --with-bugurl="$BugUrl"
                 --with-pkgversion="${AOMP_COMPILER_NAME}_$(cfgvar AOMP_VERSION_STRING)"
                 --with-gdb-datadir="\${prefix}/share/rocgdb"
                 --enable-64-bit-bfd
                 --enable-targets="x86_64-linux-gnu,amdgcn-amd-amdhsa"
                 --disable-ld --disable-gas --disable-gdbserver --disable-sim
                 --enable-tui --disable-gdbtk --disable-shared --disable-gdbtk
                 --disable-gprofng --disable-shared --with-expat
                 --with-system-zlib --without-guile --with-babeltrace
                 --with-lzma --with-python=python3
                 --with-rocm-dbgapi="$AOMP_INSTALL_DIR"
                 PKG_CONFIG_PATH="$AOMP_INSTALL_DIR/share/pkgconfig")

   mkdir -p "$BuildDir"
   export LDFLAGS="-Wl,-rpath=$AOMP_INSTALL_DIR/lib"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running gdb configure for rocgdb $Cfg ---- "
   echo "$SrcDir/configure $(shquot "${MYCONFIGOPTS[@]}")"

   if ! "$SrcDir"/configure "${MYCONFIGOPTS[@]}"; then
      echo "ERROR gdb configure failed."
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

   export CXXFLAGS_FOR_BUILD="-O2"
   export CFLAGS_FOR_BUILD="-O2"

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running make for rocgdb $Cfg ---- "
   echo "make -j $Jobs"
   if ! make -j "$Jobs"; then
      echo " "
      echo "ERROR: make -j $Jobs  FAILED"
      echo "To restart:"
      echo "  cd $BuildDir"
      echo "  make all-gdb"
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
   echo " -----Installing to $InstallDir ----- "
   echo "$SUDO make install-info-gdb"
   $SUDO make install-info-gdb
   echo "$SUDO make install-strip-gdb"

   if ! $SUDO make install-strip-gdb; then
      echo "ERROR make install failed "
      exit 1
   fi
   popd >& /dev/null || exit
}

do_list_configs() {
  echo "default"
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
