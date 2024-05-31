#!/bin/bash
#
# Wrapper script around build_supp.sh for building local/llvm-flang
# supplemental components with the flang-new driver.
#
# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

# CLEAN_SUPP=${CLEAN_SUPP:-1}    # set if clean build needed in nightly tests

cd $AOMP_REPOS
cd aomp

export AOMP_SUPP=$HOME/local/llvm-flang
export FLANG=flang-new
echo "Settings:"
echo "  AOMP=$AOMP"
echo "  AOMP_SUPP=$AOMP_SUPP"
echo "  CLEAN_SUPP=$CLEAN_SUPP"
echo "  FLANG=$FLANG"

if [ ! -r "$AOMP/bin/$FLANG" ];  then
    echo "Error: $AOMP/bin/$FLANG not present"
    exit 1
fi

if [ $CLEAN_SUPP ]; then rm -rf $AOMP_SUPP; fi
mkdir -p $AOMP_SUPP
# symlink to local prereqs that already exist, can be used
if [ ! -r $AOMP_SUPP/cmake ]; then ln -sf ../cmake $AOMP_SUPP; fi
if [ ! -r $AOMP_SUPP/ninja ]; then ln -sf ../ninja $AOMP_SUPP; fi
AOMP_USE_CCACHE=0 bin/build_supp.sh
