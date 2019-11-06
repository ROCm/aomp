#
#  setup_sim_jenkins:
#
#  setup a directory $BUILD_TEST_DIR  to simulate jenkins enviornment for building 
#  a ROCm integrated aomp.  After you run this script, you should be able to 
#  run these 4 commands from directory $BUILD_TEST_DIR to simulate a jenkins build. 
#  1. Source the environment script before running build_aomp.sh
#     . ./build/envsetup.sh
#  2. Set install location, otherwise you need root to install in /opt/rocm/aomp:
#     export AOMP=$HOME/rocmaomp/aomp
#  3. Run the build to simulate a jenkins job:
#     ./build/build_aomp.sh
#  4. Build the aomp package:
#     ./build/build_aomp.sh -p
#

GERRIT_USER=${GERRIT_USER:-$USER}
BUILD_TEST_DIR=${BUILD_TEST_DIR:-$HOME/build_test}

echo mkdir -p $BUILD_TEST_DIR
mkdir -p $BUILD_TEST_DIR

cd $BUILD_TEST_DIR
if [ -d .repo/manifests  ] ; then 
   echo cd .repo/manifests
   cd .repo/manifests
   echo git pull
   git pull
else
   mkdir .repo  
   cd .repo   
   echo git clone ssh://$GERRIT_USER@git.amd.com:29418/compute/ec/manifest.git manifests
   git clone ssh://$GERRIT_USER@git.amd.com:29418/compute/ec/manifest.git manifests
fi

cd $BUILD_TEST_DIR
if [ -d brahma-utils ] ; then 
   echo cd brahma-utils
   cd brahma-utils
   echo git pull
   git pull
else
   echo git clone ssh://$GERRIT_USER@git.amd.com:29418/brahma/ec/utility/brahma-utils
   git clone ssh://$GERRIT_USER@git.amd.com:29418/brahma/ec/utility/brahma-utils
fi

cd $BUILD_TEST_DIR
if [ -d build ] ; then 
   echo cd build
   cd build
   echo git pull
   git pull
else
   echo git clone -b jsjodin/aomp ssh://$GERRIT_USER@git.amd.com:29418/compute/ec/prototype build
   git clone -b jsjodin/aomp ssh://$GERRIT_USER@git.amd.com:29418/compute/ec/prototype build
fi
echo chmod 755 $BUILD_TEST_DIR/build/build_aomp.sh
chmod 755 $BUILD_TEST_DIR/build/build_aomp.sh

# Jenkins will use manifest to create aomp directory structure in external/aomp
# using the branches listed below.
# To simulate jenkins, we clone internal repos to external/aomp
mkdir -p $BUILD_TEST_DIR/external/aomp

COMPUTE="ssh://$GERRIT_USER@git.amd.com:29418/compute/ec"
LIGHTNING="ssh://$GERRIT_USER@git.amd.com:29418/lightning/ec"
cd $BUILD_TEST_DIR/external/aomp
if [ ! -d aomp ] ; then 
   echo git clone -b rocmaster $COMPUTE/aomp    
   git clone -b rocmaster $COMPUTE/aomp    
fi
cd $BUILD_TEST_DIR/external/aomp
if [ ! -d aomp-extras ] ; then 
   echo git clone -b rocmaster $COMPUTE/aomp-extras
   git clone -b rocmaster $COMPUTE/aomp-extras
fi
cd $BUILD_TEST_DIR/external/aomp
if [ ! -d flang ] ; then 
   echo git clone -b rocmaster $COMPUTE/flang
   git clone -b rocmaster $COMPUTE/flang
fi
cd $BUILD_TEST_DIR/external/aomp
if [ ! -d llvm-project ] ; then 
   echo git clone -b rocmaster $COMPUTE/llvm-project
   git clone -b rocmaster $COMPUTE/llvm-project
fi
cd $BUILD_TEST_DIR/external/aomp
if [ ! -d atmi ] ; then 
   echo git clone -b rocmaster $COMPUTE/atmi 
   git clone -b rocmaster $COMPUTE/atmi 
fi
cd $BUILD_TEST_DIR/external/aomp
if [ ! -d hip ] ; then 
   echo git clone -b roc-2.9.x $COMPUTE/hip
   git clone -b roc-2.9.x $COMPUTE/hip
fi
cd $BUILD_TEST_DIR/external/aomp
if [ ! -d rocminfo ] ; then 
   echo git clone -b master    $COMPUTE/rocminfo
   git clone -b master    $COMPUTE/rocminfo
fi
cd $BUILD_TEST_DIR/external/aomp
if [ ! -d rocm-compilersupport ] ; then 
   echo git clone -b aditylad/rocm-rel-2.9 $LIGHTNING/support rocm-compilersupport
   git clone -b aditylad/rocm-rel-2.9 $LIGHTNING/support rocm-compilersupport
fi
cd $BUILD_TEST_DIR/external/aomp
if [ ! -d rocm-device-libs ] ; then 
   echo git clone -b aditylad/ocl-rocm-rel-2.9 $LIGHTNING/device-libs rocm-device-libs
   git clone -b aditylad/ocl-rocm-rel-2.9 $LIGHTNING/device-libs rocm-device-libs
fi

cd $BUILD_TEST_DIR
