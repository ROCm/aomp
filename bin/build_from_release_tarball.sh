#!/bin/bash
# 
#  build_from_release_tarball.sh: 
#
#  Starting with aomp release 0.7-5, a complete source tarball of all necessary repos is built as 
#  a release artifact. This tarball already has the rocm patches applied.  So we turn off AOMP_APPLY_ROCM_PATCHES.
#  Since the expanded tarball contains no git information, we must turn off AOMP_CHECK_GIT_BRANCH. 
#

# These four values determine where the build scripts will install AOMP
AOMP_VERSION=${AOMP_VERSION:-"0.7"}
AOMP_VERSION_MOD=${AOMP_VERSION_MOD:-"5"}
AOMP_VERSION_STRING=${AOMP_VERSION_STRING:-"$AOMP_VERSION-$AOMP_VERSION_MOD"}
# The build scripts will install to $AOMP-$AOMP_$AOMP_VERSION_STRING .
# $AOMP will become a symbolic link to the versioned install directory .
# Default value of $AOMP if not specified by caller's environment is /opt/rocm/aomp
AOMP=${AOMP:-/opt/rocm/aomp}

# Make the TMPDIR, download the tarball, and expand the tarball sources
TMPDIR=/tmp/$USER/aomp_${AOMP_VERSION_STRING}
mkdir -p $TMPDIR
cd $TMPDIR
tarball=aomp_${AOMP_VERSION_STRING}.tar.gz
release_url="https://github.com/ROCm-Developer-Tools/aomp/releases/download"
wget $release_url/rel_${AOMP_VERSION_STRING}/$tarball 2>/dev/null
if [ $? != 0 ] ; then 
   echo
   echo "$0: failed to wget the release tarball $tarball"
   echo "from $release_url/rel_${AOMP_VERSION_STRING}/$tarball"
   echo
   exit 1
fi
tar -xzf aomp_${AOMP_VERSION_STRING}.tar.gz
if [ $? != 0 ] ; then 
   echo
   echo "$0: failed to unpack $tarball"
   echo
   exit 1
fi


# set environment values that build_aomp.sh will use
AOMP_REPOS=$TMPDIR/aomp
AOMP_CHECK_GIT_BRANCH=0
AOMP_APPLY_ROCM_PATCHES=0
export AOMP AOMP_CHECK_GIT_BRANCH AOMP_APPLY_ROCM_PATCHES AOMP_VERSION AOMP_VERSION_MOD AOMP_VERSION_STRING AOMP_REPOS

# no need to relocate sources to build directories.
unset BUILD_AOMP

# Call the master build script that will build each component
$TMPDIR/aomp/aomp/bin/build_aomp.sh
