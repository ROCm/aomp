#!/bin/bash
#
#  File: build_libm.sh
#        buind the rocm-device-libs libraries in $AOMP/lib/libm/$MCPU
#        The rocm-device-libs get built for each processor in $GFXLIST
#        even though currently all rocm-device-libs are identical for each 
#        gfx processor (amdgcn)
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
[ ! -L "$0" ] || thisdir=$(getdname `readlink "$0"`)
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_DIR=${INSTALL_LIBM:-"${AOMP_INSTALL_DIR}"}

LIBM_DIR_SRC="$AOMP_REPOS/$AOMP_REPO_NAME/examples/libdevice/libm"
LIBM_DIR="$BUILD_DIR/build/libm"
echo rsync -av $LIBM_DIR_SRC/ $LIBM_DIR/
rsync -av $LIBM_DIR_SRC/ $LIBM_DIR/

export PATH=$LLVM_BUILD/bin:$PATH

if [ "$1" != "install" ] ; then 
   savecurdir=$PWD
   cd $LIBM_DIR
   make clean-out
   for gpu in $GFXLIST ; do
      echo  
      echo "BUilding libm for $gpu"
      echo
      AOMP_GPU=$gpu make
      if [ $? != 0 ] ; then
	 echo 
         echo "ERROR make failed for AOMP_GPU = $gpu"
         exit 1
      fi
   done
   PROC=`uname -p`
   if [ "$PROC" == "aarch64" ] ; then
      echo "WARNING:  No cuda for aarch64 so skipping libm creation for $NVPTXGPUS"
   else
      origIFS=$IFS
      IFS=","
      for gpu in $NVPTXGPUS ; do
         echo  
         echo "BUilding libm for sm_$gpu"
         echo
         AOMP_GPU="sm_$gpu" make
         if [ $? != 0 ] ; then
	    echo 
            echo "ERROR make failed for AOMP_GPU = sm_$gpu"
            exit 1
         fi
      done
      IFS=$origIFS
   fi
   cd $savecurdir
   echo 
   echo "  Done! Please run ./build_libm.sh install "
   echo 
fi

if [ "$1" == "install" ] ; then 
   echo
   echo "INSTALLING DBCL libm from $LIBM_DIR/build "
   echo "rsync -av $LIBM_DIR/build/libdevice $INSTALL_DIR/lib"
   mkdir -p $INSTALL_DIR/lib
   $SUDO rsync -av $LIBM_DIR/build/libdevice $INSTALL_DIR/lib
   echo "rsync -av $LIBM_DIR/build/libdevice $INSTALL_DIR/lib-debug"
   mkdir -p $INSTALL_DIR/lib-debug
   $SUDO rsync -av $LIBM_DIR/build/libdevice $INSTALL_DIR/lib-debug
   echo 
   echo " $0 Installation complete into $INSTALL_DIR"
   echo 
fi
