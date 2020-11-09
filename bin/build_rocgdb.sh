#!/bin/bash
#
#  build_rocgdb.sh:  Script to build ROCgdb for aomp standalone build
#                 This will be called by build_aomp.sh when
#                 AOMP_STANDALONE_BUILD=1 && AOMP_BUILD_DEBUG==1
#                 This depends on rocdbgapi to be built and installed.
#
# --- Start standard header ----
function getdname(){
   local __DIRN=`dirname "$1"`
   if [ "$__DIRN" = "." ] ; then
      __DIRN=$PWD;
   else
      if [ ${__DIRN:0:1} != "/" ] ; then
         if [ ${__DIRN:0:2} == ".." ] ; then
               __DIRN=`dirname $PWD`/${__DIRN:3}
         else
            if [ ${__DIRN:0:1} = "." ] ; then
               __DIRN=$PWD/${__DIRN:2}
            else
               __DIRN=$PWD/$__DIRN
            fi
         fi
      fi
   fi
   echo $__DIRN
}

thisdir=$(getdname $0)
. $thisdir/aomp_common_vars
# --- end standard header ----

# Point to the right python3.6 on Red Hat 7.6
if [ -f /opt/rh/rh-python36/enable ]; then
  export PATH=/opt/rh/rh-python36/root/usr/bin:$PATH
  export LIBRARY_PATH=/opt/rh/rh-python36/root/lib64:$LIBRARY_PATH
fi

REPO_DIR=$AOMP_REPOS/$AOMP_GDB_REPO_NAME
REPO_BRANCH=$AOMP_GDB_REPO_BRANCH
checkrepo

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds ROCgdb for AOMP standalone build"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_GDB_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/rocgdb"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocgdb.sh                   configure, make , NO Install "
  echo "  ./build_rocgdb.sh noconfigure       NO configure, make, NO install "
  echo "  ./build_rocgdb.sh install           NO configure, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/$AOMP_GDB_REPO_NAME ] ; then 
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_GDB_REPO_NAME"
   echo "        Are environment variables AOMP_REPOS and AOMP_GDB_REPO_NAME set correctly?"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $AOMP_INSTALL_DIR
   $SUDO touch $AOMP_INSTALL_DIR/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $AOMP_INSTALL_DIR"
      exit 1
   fi
   $SUDO rm $AOMP_INSTALL_DIR/testfile
fi

BUG_URL="https://github.com/ROCm-Developer-Tools/ROCgdb/issues"
export CXXFLAGS_FOR_BUILD="-O2"
export CFLAGS_FOR_BUILD="-O2"
#patchrepo $AOMP_REPOS/$AOMP_GDB_REPO_NAME
if [ "$1" != "noconfigure" ] && [ "$1" != "install" ] ; then 
   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_rocgdb"
   echo "Use ""$0 noconfigure"" or ""$0 install"" to avoid FRESH START."
   BUILDTYPE="Release"
   echo rm -rf $BUILD_AOMP/build/rocgdb
   rm -rf $BUILD_AOMP/build/rocgdb
   MYCONFIGOPTS="--prefix=$AOMP_INSTALL_DIR --srcdir=$AOMP_REPOS/$AOMP_GDB_REPO_NAME --program-prefix=roc \
     --with-bugurl="$BUG_URL" --with-pkgversion="${AOMP_COMPILER_NAME}_${AOMP_VERSION_STRING}" \
     --enable-targets="x86_64-linux-gnu,amdgcn-amd-amdhsa" \
     --disable-ld --disable-gas --disable-gdbserver --disable-sim --enable-tui \
     --disable-gdbtk --disable-shared  \
     --with-expat --with-system-zlib --without-guile --with-babeltrace --with-lzma \
     --with-python=python3 --with-rocm-dbgapi=$AOMP_INSTALL_DIR"
   mkdir -p $BUILD_AOMP/build/rocgdb
   cd $BUILD_AOMP/build/rocgdb
   echo " -----Running gdb configure ---- " 
   echo "$AOMP_REPOS/$AOMP_GDB_REPO_NAME/configure $MYCONFIGOPTS"
   $AOMP_REPOS/$AOMP_GDB_REPO_NAME/configure $MYCONFIGOPTS
   if [ $? != 0 ] ; then 
      echo "ERROR gdb configure failed."
      exit 1
   fi
fi

cd $BUILD_AOMP/build/rocgdb
echo
echo " -----Running make for gdb ---- " 
echo make -j $NUM_THREADS all-gdb
make -j $NUM_THREADS all-gdb
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $NUM_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/rocgdb"
      echo "  make all-gdb"
      exit 1
else
   if [ "$1" != "install" ] ; then
      echo
      echo "Successful build of ./build_rocgdb.sh .  Please run:"
      echo "  ./build_rocgdb.sh install "
      echo "to install into directory $AOMP_INSTALL_DIR"
      echo
   fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/rocgdb
      echo " -----Installing to $AOMP_INSTALL_DIR ----- " 
      echo $SUDO make install-info-gdb
      $SUDO make install-info-gdb
      echo $SUDO make install-strip-gdb
      $SUDO make install-strip-gdb
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
#     removepatch $AOMP_REPOS/$AOMP_GDB_REPO_NAME
fi
