#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
#  prebuild_srock.sh: Source this file from build_srock.sh
#     update the srock repo
#     clone or update hipfort repo
#     builds cmake if necessary in ~/local/cmake using
#
# --- Start standard header to set SROCK environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/srock_common_vars"
# --- end standard header ----

echo "= 1 = Checking if cmake needs to be built"
if [ $_build_cmake == 1 ] ; then
   echo "      building $_cmake_local"
   $thisdir/build_cmake.sh
else
   echo "      using existing $_cmake_local"
fi
echo "      The cmake for srock is $SROCK_CMAKE"

# Skip these updates if this is a restart
if [ "$_build_srock_mode" == "restart" ] ; then
   return
fi

cd $SROCK_REPOS
echo "= 2 = Updating aomp repo"
if [ -d $SROCK_REPOS/aomp ] ; then
   echo "      Skipping aomp clone, $SROCK_REPOS/srock already exists"
else
   echo "      git clone -b $SROCK_DEV_BRANCH https://github.com/ROCm/aomp"
   git clone -b $SROCK_DEV_BRANCH  https://github.com/ROCm/aomp 2>/dev/null >/dev/null
fi
echo "      cd $SROCK_REPOS/aomp"
cd $SROCK_REPOS/aomp
echo "      git checkout $SROCK_DEV_BRANCH"
git checkout $SROCK_DEV_BRANCH
echo "      git pull"
git pull

cd $SROCK_REPOS
echo "= 3 = Updating hipfort repo"
if [ -d $SROCK_REPOS/hipfort ] ; then
   echo "      Skipping hipfort clone, $SROCK_REPOS/hipfort already exists"
else
   echo "      git clone -b $SROCK_HIPFORT_BRANCH https://github.com/ROCm/hipfort"
   git clone -b $SROCK_HIPFORT_BRANCH  https://github.com/ROCm/hipfort 2>/dev/null >/dev/null
fi
echo "      cd $SROCK_REPOS/hipfort"
cd $SROCK_REPOS/hipfort
echo "      git checkout $SROCK_HIPFORT_BRANCH"
git checkout $SROCK_HIPFORT_BRANCH
echo "      git pull"
git pull

