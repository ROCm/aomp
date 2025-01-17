#!/bin/bash
#
#  File: build_extras.sh
#        Modify and copy former aomp-extras (now in aomp) utilities to aomp install.
#        The install option will install components into the aomp installation.
#        Note: this script does not use cmake or make steps.
#
# MIT License
#
# Copyright (c) 2019 Advanced Micro Devices, Inc. All Rights Reserved.
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

AOMP_REPO_DIR=$AOMP_REPOS/$AOMP_REPO_NAME

BUILD_DIR=${BUILD_AOMP}

INSTALL_EXTRAS=${INSTALL_EXTRAS:-$LLVM_INSTALL_LOC}
export LLVM_DIR=$LLVM_INSTALL_LOC

if [ "$AOMP_STANDALONE_BUILD" == 1 ] ; then
  install_list="gpurun rebundle_hip_lib.sh raja_build.sh kokkos_build.sh aompversion blt.patch raja.patch modulefile"
else
  install_list="gpurun"
fi

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_extras.sh                   copy to build location, NO Install "
  echo "  ./build_extras.sh install           install "
  echo " "
  exit
fi

# Make sure we can update the install directory
check_writable_installdir "$1" "$INSTALL_EXTRAS"

if [ "$1" != "install" ] ; then
  if [ -d "$BUILD_DIR/build/extras" ] ; then
    echo
    echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
    echo "rm -rf $BUILD_DIR/build/extras"
    rm -rf "$BUILD_DIR/build/extras"
  fi

  if [ "$AOMP_STANDALONE_BUILD" == 0 ] ; then
    export AOMP_VERSION_STRING=$ROCM_VERSION
  fi

  mkdir -p "$BUILD_DIR/build/extras"
  cd "$BUILD_DIR/build/extras" || exit

  if [ "$AOMP_STANDALONE_BUILD" == 0 ] ; then
    SED_INSTALL_DIR=$(echo /opt/rocm/llvm | sed -e 's/\//\\\\\//g')
  else
    SED_INSTALL_DIR=$(echo "$INSTALL_EXTRAS" | sed -e 's/\//\\\\\//g')
  fi

  export SED_INSTALL_DIR

  echo "----- Copy util scripts to $BUILD_DIR/build/extras -----"
  cp "$AOMP_REPO_DIR"/utils/* "$BUILD_DIR"/build/extras

  for util in $install_list; do
    if [ "$util" == "rebundle_hip_lib.sh" ]; then
      /bin/sed -i -e "s/X\\.Y\\-Z/${AOMP_VERSION_STRING}/g" -e "s/_LLVM_INSTALL_DIR_/${SED_INSTALL_DIR}/g" "$util"
    else
      /bin/sed -i -e "s/X\\.Y\\-Z/${AOMP_VERSION_STRING}/g" -e "s/_AOMP_INSTALL_DIR_/${SED_INSTALL_DIR}/g" "$util"
    fi
  done
fi

cd "$BUILD_DIR/build/extras" || exit
echo
if [ "$1" != "install" ] ; then
  echo
  echo " BUILD COMPLETE! To install extras component run this command:"
  echo "  $0 install"
  echo
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
  cd "$BUILD_DIR/build/extras" || exit
  echo " -----Installing to $INSTALL_EXTRAS/bin ----- "
  for util in $install_list; do
    echo "-- Installing: $INSTALL_EXTRAS/bin/$util"
    cp "$BUILD_DIR"/build/extras/"$util" "$INSTALL_EXTRAS"/bin
    echo "$INSTALL_EXTRAS/bin/$util" >> install_manifest.txt
  done
  if [ "$AOMP_STANDALONE_BUILD" == 1 ] ; then
    if [ -f "$LLVM_INSTALL_LOC/bin/gpurun" ] && [ ! -h "$AOMP_INSTALL_DIR/bin/gpurun" ]; then
      echo "Creating gpurun symlink: ${AOMP_INSTALL_DIR}/bin/gpurun -> ${LLVM_INSTALL_LOC}/bin/gpurun"
      ln -s ../lib/llvm/bin/gpurun "$AOMP_INSTALL_DIR"/bin/gpurun
    fi
  fi
fi
