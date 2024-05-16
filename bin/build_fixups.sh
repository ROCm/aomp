#!/bin/bash
#
#   build_fixups.sh : make some fixes to the installation.
#                     We eventually need to remove this hack.
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

# Copy examples 
if [ -d $AOMP/examples ] ; then 
  $SUDO rm -rf $AOMP/examples
fi
echo $SUDO cp -rp $AOMP_REPOS/$AOMP_REPO_NAME/examples $AOMP
$SUDO cp -rp $AOMP_REPOS/$AOMP_REPO_NAME/examples $AOMP

if [ "$AOMP_STANDALONE_BUILD" == 1 ] ; then
  # Licenses
  echo mkdir -p $AOMP/share/doc/aomp
  mkdir -p $AOMP/share/doc/aomp
  echo $SUDO cp $AOMP_REPOS/$AOMP_REPO_NAME/LICENSE $AOMP/share/doc/aomp/LICENSE.apache2
  $SUDO cp $AOMP_REPOS/$AOMP_REPO_NAME/LICENSE $AOMP/share/doc/aomp/LICENSE.apache2
  echo $SUDO cp $AOMP_REPOS/$AOMP_EXTRAS_REPO_NAME/LICENSE $AOMP/share/doc/aomp/LICENSE.mit
  $SUDO cp $AOMP_REPOS/$AOMP_EXTRAS_REPO_NAME/LICENSE $AOMP/share/doc/aomp/LICENSE.mit
  echo $SUDO cp $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/LICENSE.txt $AOMP/share/doc/aomp/LICENSE.flang
  $SUDO cp $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/LICENSE.txt $AOMP/share/doc/aomp/LICENSE.flang
fi

echo Cleaning AOMP Directory...
#examples
$SUDO rm -f $AOMP/examples/hip/*.txt
$SUDO rm -f $AOMP/examples/hip/*.sh
$SUDO rm -f $AOMP/examples/openmp/*.txt
$SUDO rm -f $AOMP/examples/openmp/*.sh
$SUDO rm -f $AOMP/examples/cloc/*.txt
$SUDO rm -f $AOMP/examples/cloc/*.sh
$SUDO rm -f $AOMP/examples/fortran/*.txt
$SUDO rm -f $AOMP/examples/fortran/*.sh
$SUDO rm -f $AOMP/examples/*.sh
$SUDO rm -f $AOMP/examples/raja/*.txt
$SUDO rm -f $AOMP/examples/raja/*.sh

# Clean libexec, for now just delete files not directories
# rocprofiler installs some needed python scripts in
# libexec/rocprofiler.
$SUDO find $AOMP/libexec -maxdepth 1 -type f -delete

# Clean src
$SUDO rm -rf $AOMP/src
$SUDO rm -rf $AOMP/rocclr

# Clean llvm-lit
$SUDO rm -f $AOMP/bin/llvm-lit

echo "Done with $0"
