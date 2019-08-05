#!/bin/bash
# 
#   build_aomp.sh : Build all AOMP components 
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

function build_aomp_component() {
   $AOMP_REPOS/$AOMP_REPO_NAME/bin/build_$COMPONENT.sh "$@"
   rc=$?
   if [ $rc != 0 ] ; then 
      echo " !!!  build_aomp.sh: BUILD FAILED FOR COMPONENT $COMPONENT !!!"
      exit $rc
   fi  
   if [ $# -eq 0 ] ; then
       $AOMP_REPOS/$AOMP_REPO_NAME/bin/build_$COMPONENT.sh install
       rc=$?
       if [ $rc != 0 ] ; then 
           echo " !!!  build_aomp.sh: INSTALL FAILED FOR COMPONENT $COMPONENT !!!"
           exit $rc
       fi
   fi
}


TOPSUDO=${SUDO:-NO}
if [ "$TOPSUDO" == "set" ]  || [ "$TOPSUDO" == "yes" ] || [ "$TOPSUDO" == "YES" ] ; then
   TOPSUDO="sudo"
else
   TOPSUDO=""
fi

# Test update access to AOMP_INSTALL_DIR
# This should be done early to ensure sudo (if set) does not prompt for password later
$TOPSUDO mkdir -p $AOMP_INSTALL_DIR
if [ $? != 0 ] ; then
   echo "ERROR: $TOPSUDO mkdir failed, No update access to $AOMP_INSTALL_DIR"
   exit 1
fi
$TOPSUDO touch $AOMP_INSTALL_DIR/testfile
if [ $? != 0 ] ; then
   echo "ERROR: $TOPSUDO touch failed, No update access to $AOMP_INSTALL_DIR"
   exit 1
fi
$TOPSUDO rm $AOMP_INSTALL_DIR/testfile

echo 
date
echo " =================  START build_aomp.sh ==================="   
echo 

if [ "$AOMP_BUILD_HIP" == 1 ] ; then
   components="roct rocr project libdevice comgr hcc hip extras atmi openmp pgmath flang flang_runtime"
else
   # The hip build will only install headers if AOMP_BUILD_HIP is off
   # if AOMP_BUILD_HIP is off, then hcc is not built
   components="roct rocr project libdevice comgr hip extras atmi openmp pgmath flang flang_runtime"
fi
for COMPONENT in $components ; do 
   echo 
   echo " =================  BUILDING COMPONENT $COMPONENT ==================="   
   echo 
   build_aomp_component "$@"
   date
   echo " =================  DONE INSTALLING COMPONENT $COMPONENT ==================="   
done
#Run build_fixups.sh to clean the AOMP directory before packaging
#$AOMP_REPOS/$AOMP_REPO_NAME/bin/build_fixups.sh
echo 
date
echo " =================  END build_aomp.sh ==================="   
echo 
exit 0
