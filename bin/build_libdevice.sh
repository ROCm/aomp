#!/bin/bash
#
#  File: build_libdevice.sh
#        build the rocm-device-libs libraries in $AOMP/lib/libdevice
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

# We now pickup HSA from the AOMP install directory because it is built
# with build_roct.sh and build_rocr.sh . 
HSA_DIR=${HSA_DIR:-$AOMP}
SKIPTEST=${SKIPTEST:-"YES"}

INSTALL_ROOT_DIR=${INSTALL_LIBDEVICE:-$AOMP_INSTALL_DIR}
INSTALL_DIR=$INSTALL_ROOT_DIR

export LLVM_DIR=$AOMP_INSTALL_DIR
export LLVM_BUILD=$AOMP_INSTALL_DIR
SOURCEDIR=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_LIBDEVICE_REPO_NAME

REPO_DIR=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_LIBDEVICE_REPO_NAME

MYCMAKEOPTS="-DLLVM_DIR=$LLVM_DIR -DCMAKE_INSTALL_LIBDIR=lib"

if [ ! -d $AOMP_INSTALL_DIR/lib ] ; then 
  echo "ERROR: Directory $AOMP/lib is missing"
  echo "       AOMP must be installed in $AOMP_INSTALL_DIR to continue"
  exit 1
fi

export LLVM_BUILD HSA_DIR
export PATH=$LLVM_BUILD/bin:$PATH

patchrepo $REPO_DIR

if [ "$1" != "install" ] ; then 
    
      builddir_libdevice=$BUILD_DIR/build/libdevice
      if [ -d $builddir_libdevice ] ; then 
         echo rm -rf $builddir_libdevice
         # need SUDO because a previous make install was done with sudo 
         $SUDO rm -rf $builddir_libdevice
      fi
      mkdir -p $builddir_libdevice
      cd $builddir_libdevice
      echo 
      echo DOING BUILD in Directory $builddir_libdevice
      echo 

      CC="$LLVM_BUILD/bin/clang"
      export CC
      echo "${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_LIBDEVICE_REPO_NAME"
      ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/$AOMP_LIBDEVICE_REPO_NAME
      if [ $? != 0 ] ; then 
         echo "ERROR cmake failed  command was \n"
         echo "      ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR $BUILD_DIR/$AOMP_LIBDEVICE_REPO_NAME"
         exit 1
      fi
      echo "make -j $AOMP_JOB_THREADS"
      make -j $AOMP_JOB_THREADS 
      if [ $? != 0 ] ; then 
         echo "ERROR make failed "
         exit 1
      fi


   echo 
   echo "  Done with all makes"
   echo "  Please run ./build_libdevice.sh install "
   echo 

   if [ "$SKIPTEST" != "YES" ] ; then 
         builddir_libdevice=$BUILD_DIR/build/libdevice
         cd $builddir_libdevice
         echo "running tests in $builddir_libdevice"
         make test 
      echo 
      echo "# done with all tests"
      echo 
   fi
fi

if [ "$1" == "install" ] ; then 

   echo 
   echo mkdir -p $INSTALL_DIR/include
   $SUDO mkdir -p $INSTALL_DIR/include
   $SUDO mkdir -p $INSTALL_DIR/lib
   builddir_libdevice=$BUILD_DIR/build/libdevice
   echo "running make install from $builddir_libdevice"
   cd $builddir_libdevice
   echo $SUDO make -j $AOMP_JOB_THREADS install
   $SUDO make -j $AOMP_JOB_THREADS install

   # rocm-device-lib cmake installs to lib dir, move all bc files up one level
   # and cleanup unused oclc_isa_version bc files and link correct one
   #echo
   #echo "POST-INSTALL REORG OF SUBDIRECTORIES $INSTALL_DIR"
   #echo "--"
   #echo "-- $INSTALL_DIR"
   #echo "-- MOVING bc FILES FROM lib DIRECTORY UP ONE LEVEL"
   #$SUDO mv $INSTALL_DIR/lib/*.bc $INSTALL_DIR
   #$SUDO rm -rf $INSTALL_DIR/lib/cmake
   #$SUDO rmdir $INSTALL_DIR/lib 

   # END OF POST-INSTALL REORG 

   echo 
   echo " $0 Installation complete into $INSTALL_DIR"
   echo 
   removepatch $REPO_DIR
fi
