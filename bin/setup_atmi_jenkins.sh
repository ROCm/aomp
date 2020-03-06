#!/bin/bash

GERRIT_USER=${GERRIT_USER:-$USER}
BUILD_ATMI_DIR=${BUILD_ATMI_DIR:-$HOME/build_atmi_jenkins}

echo mkdir -p $BUILD_ATMI_DIR
mkdir -p $BUILD_ATMI_DIR
cd $BUILD_ATMI_DIR
if [ -d brahma-utils ] ; then
   echo cd brahma-utils
   cd brahma-utils
   echo git pull
   git pull
else
   echo git clone ssh://$GERRIT_USER@git.amd.com:29418/brahma/ec/utility/brahma-utils
   git clone ssh://$GERRIT_USER@git.amd.com:29418/brahma/ec/utility/brahma-utils
fi
cd $BUILD_ATMI_DIR
if [ -d .repo/manifests  ] ; then 
   echo cd .repo/manifests
   cd .repo/manifests
   echo git pull
   git pull
else
   repo init -u ssh://gerritgit/compute/ec/manifest.git -m compute.xml
   cd .repo/manifests
   echo "checkout master-no-npi"
   git checkout master-no-npi
   echo "repo sync"
fi

repo sync lightning/ec/llvm-project lightning/ec/device-libs lightning/ec/support compute/ec/atmi lightning/ec/support hsa/ec/hsa-runtime compute/ec/libhsakmt -d -c -q --force-sync --jobs=8

cd $BUILD_ATMI_DIR
if [ -d build ] ; then 
   echo cd build
   cd build
   echo git pull
   git pull
else
   echo git clone -b amd-master ssh://$GERRIT_USER@git.amd.com:29418/compute/ec/prototype build
   git clone -b amd-master ssh://$GERRIT_USER@git.amd.com:29418/compute/ec/prototype build
fi

cd $BUILD_ATMI_DIR


