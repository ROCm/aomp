#!/bin/bash
#
#  build_rocm_sysdeps.sh:  Script to build the rocm runtime and install into the
#                  aomp compiler installation
#                  Requires that "build_roct.sh install" be installed first
#

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/aomp_common_vars"
# --- end standard header ----

set -eo pipefail
set -E

# Error Handling
cmd_error(){
  echo "Error: build_rocm_sysdeps.sh failed"
  exit 1
}

trap cmd_error ERR

INSTALL_ROCM_SYSDEPS=${INSTALL_ROCM_SYSDEPS:-$AOMP_INSTALL_DIR/rocm_sysdeps}
ROCM_SYSDEPS_LIST=${ROCM_SYSDEPS_LIST:-libdrm}
BUILD_DIR=$BUILD_AOMP/build/rocm_sysdeps

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo " This script builds the ROCM sysdeps libraries"
  echo " It gets the source from:  Functions in this script"
  echo " It builds libraries in:   $BUILD_AOMP/build/rocm_sysdeps"
  echo " It installs in:           $INSTALL_ROCM_SYSDEPS"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocm_sysdeps.sh                   cmake, make , NO Install "
  echo "  ./build_rocm_sysdeps.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit
fi

check_writable_installdir "$1" "$INSTALL_ROCM_SYSDEPS"

###############################################################################################
# TheRock ships libdrm in rocm_sysdeps and therefore will not invoke a system install of libdrm.
# AOMP needs this library shipped with the install to satisfy dlopen of libhsa.
###############################################################################################

function buildlibdrm(){
  _cname="libdrm"
  _version="2.4.134"
  _installdir="$INSTALL_ROCM_SYSDEPS"
  _builddir="$BUILD_DIR/$_cname"
  _patcheddir="$BUILD_DIR/$_cname-patched"
  _srcdir="$_builddir/$_cname-$_cname-$_version"
  TARBALL_URL="https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/libdrm-libdrm-${_version}.tar.bz2"
  THEROCK=https://raw.githubusercontent.com/ROCm/TheRock/main/third-party/sysdeps/linux/libdrm
  AMDGPU_IDS="$_patcheddir/data/amdgpu.ids"

  if [ "$1" != "install" ]; then
    if [ -d "$_builddir" ] ; then
      rm -rf "$_builddir"
    fi
    mkdir -p "$_builddir"
    cd "$_builddir"

    wget $THEROCK/inline_amdgpu_ids.sh
    chmod +x inline_amdgpu_ids.sh
    wget "$TARBALL_URL"
    tar -xf $_cname-$_cname-$_version.tar.bz2
    # Need to copy to patch directory or inline_amdgpu_ids.sh complains about patching source
    cp -a "$_srcdir" "$_patcheddir"

    # Patch libdrm source with additional amdgpu ids
    cat >> "$AMDGPU_IDS" << 'EOF'
7590,   C0,     AMD Radeon RX 9060 XT
7590,   CF,     AMD Radeon RX 9050
7590,   DF,     AMD Radeon RX 9050 4GB
744B,   00,     AMD Radeon PRO W7900D
75A0,   00,     AMD Instinct MI350X
75A3,   00,     AMD Instinct MI355X
75B0,   00,     AMD Instinct MI350X VF
75B3,   00,     AMD Instinct MI355X VF
75A8,   00,     AMD Instinct MI350P
7551,   C1,     AMD Radeon AI Pro R9700S
7551,   C8,     AMD Radeon AI Pro R9600D
EOF

    "$_builddir"/inline_amdgpu_ids.sh "$_patcheddir"
    cd "$_cname-$_cname-$_version"
    # Configure libdrm
    meson setup "$_builddir" "$_patcheddir" \
    --prefix "$_installdir" \
    -Dpkgconfig.relocatable=true \
    -Dlibdir=lib \
    -Damdgpu=enabled \
    -Dintel=disabled \
    -Dman-pages=disabled \
    -Dnouveau=disabled \
    -Dradeon=disabled \
    -Dvmwgfx=disabled

    # Build libdrm
    meson compile -C "$_builddir" --verbose
  fi
  if [ "$1" == "install" ]; then
    meson install -C "$_builddir"
  fi
}

function main(){
  for _component in $ROCM_SYSDEPS_LIST ; do
    if [ "$_component" == "libdrm" ] ; then
    buildlibdrm "$1"
  else
    echo "ERROR:  Invalid component name $_component"
    exit 1
  fi
done
}

if [ "$1" != "install" ] ; then
  echo " "
  echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR"
  echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
  echo "rm -rf $BUILD_DIR"
  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR" || exit
  echo

  echo " -----Running rocm_sysdeps cmake ---- "
  # Call main function
  main
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
  cd "$BUILD_DIR" || exit
  echo " -----Installing to $INSTALL_ROCM_SYSDEPS/lib ----- "
  # Call main function install
  main install
  # hwloc depends on rocmsmilib due to (--with-rocm) build option. rocmsmilib depends on libdrm. Moving hwloc and rocmsmilib build invocation here.
  ROCM_SYSDEPS_PATH=$INSTALL_ROCM_SYSDEPS PREREQUISITE_COMPONENTS="rocmsmilib hwloc" "$thisdir/build_prereq.sh"
fi

