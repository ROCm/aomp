#!/bin/bash
#
#  build_gdb.sh:  Script to build gdb
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

REPO_DIR=$AOMP_REPOS/$AOMP_GDB_REPO_NAME
REPO_BRANCH=$AOMP_GDB_REPO_BRANCH
checkrepo

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds GDB"
  echo " It gets the source from:  $AOMP_REPOS/$AOMP_GDB_REPO_NAME"
  echo " It builds libraries in:   $BUILD_AOMP/build/gdb"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_gdb.sh                   configure, make , NO Install "
  echo "  ./build_gdb.sh noconfigure       NO configure, make, NO install "
  echo "  ./build_gdb.sh install           NO configure, make , INSTALL"
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

#patchrepo $AOMP_REPOS/$AOMP_GDB_REPO_NAME
if [ "$1" != "noconfigure" ] && [ "$1" != "install" ] ; then 
   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build_gdb"
   echo "Use ""$0 noconfigure"" or ""$0 install"" to avoid FRESH START."
   BUILDTYPE="Release"
   echo rm -rf $BUILD_AOMP/build/gdb
   rm -rf $BUILD_AOMP/build/gdb
   MYCONFIGOPTS="--prefix=$AOMP_INSTALL_DIR/gdb  --srcdir=$AOMP_REPOS/$AOMP_GDB_REPO_NAME"
   mkdir -p $BUILD_AOMP/build/gdb
   cd $BUILD_AOMP/build/gdb
   echo " -----Running gdb configure ---- " 
   $AOMP_REPOS/$AOMP_GDB_REPO_NAME/configure $MYCONFIGOPTS
   if [ $? != 0 ] ; then 
      echo "ERROR gdb configure failed."
      exit 1
   fi
fi

cd $BUILD_AOMP/build/gdb
echo
echo " -----Running make for gdb ---- " 
echo make -j $NUM_THREADS all-gdbsupport all-gdb
make -j $NUM_THREADS all-gdbsupport all-gdb
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $NUM_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/gdb"
      echo "  make"
      exit 1
else
   if [ "$1" != "install" ] ; then
      echo
      echo "Successful build of ./build_gdb.sh .  Please run:"
      echo "  ./build_gdb.sh install "
      echo "to install into directory $AOMP_INSTALL_DIR/gdb"
      echo
   fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/gdb
      echo " -----Installing to $AOMP_INSTALL_DIR/gdb ----- " 
      echo $SUDO make install-gdb
      $SUDO make install-gdb
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
#     removepatch $AOMP_REPOS/$AOMP_GDB_REPO_NAME
fi
