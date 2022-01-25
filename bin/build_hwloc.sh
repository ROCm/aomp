#!/bin/bash
#
#  build_hwloc.sh:  Script to build hwloc for aomp standalone build
#                 This will be called by build_aomp.sh when
#                 AOMP_STANDALONE_BUILD=1
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

REPO_DIR=$AOMP_REPOS/hwloc

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds hwloc for AOMP standalone build"
  echo " It gets the source from:  $AOMP_REPOS/hwloc"
  echo " It builds libraries in:   $BUILD_AOMP/build/hwloc"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_hwloc.sh                   configure, make , NO Install "
  echo "  ./build_hwloc.sh noconfigure       NO configure, make, NO install "
  echo "  ./build_hwloc.sh install           NO configure, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 
fi

if [ ! -d $AOMP_REPOS/hwloc ] ; then 
   echo "ERROR:  Missing repository $AOMP_REPOS/hwloc"
   echo "        Is environment variables AOMP_REPOS set correctly?"
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

export CXXFLAGS_FOR_BUILD="-O2"
export CFLAGS_FOR_BUILD="-O2"
if [ "$1" != "noconfigure" ] && [ "$1" != "install" ] ; then 
   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_AOMP/build/hwloc"
   echo "Use ""$0 noconfigure"" or ""$0 install"" to avoid FRESH START."
   echo rm -rf $BUILD_AOMP/build/hwloc
   rm -rf $BUILD_AOMP/build/hwloc
   MYCONFIGOPTS="--prefix=$AOMP_INSTALL_DIR --srcdir=$AOMP_REPOS/hwloc"
   mkdir -p $BUILD_AOMP/build/hwloc
   cd $BUILD_AOMP/build/hwloc
   echo
   echo " -----Running autogen ---- " 
   $AOMP_REPOS/hwloc/autogen.sh
   echo
   echo " -----Running hwloc configure ---- " 
   echo "$AOMP_REPOS/hwloc/configure $MYCONFIGOPTS"
   $AOMP_REPOS/hwloc/configure $MYCONFIGOPTS
   if [ $? != 0 ] ; then 
      echo "ERROR hwloc configure failed."
      exit 1
   fi
fi

cd $BUILD_AOMP/build/hwloc
echo
echo " -----Running make for hwloc ---- " 
echo make -j $AOMP_JOB_THREADS
make -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_AOMP/build/hwloc"
      echo "  make"
      exit 1
else
   if [ "$1" != "install" ] ; then
      echo
      echo "Successful build of ./build_hwloc.sh .  Please run:"
      echo "  ./build_hwloc.sh install "
      echo "to install into directory $AOMP_INSTALL_DIR"
      echo
   fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
      cd $BUILD_AOMP/build/hwloc
      echo " -----Installing to $AOMP_INSTALL_DIR ----- " 
      echo $SUDO make install
      $SUDO make install
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi
fi
