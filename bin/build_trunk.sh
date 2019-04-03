#!/bin/bash
# 
#   build_trunk.sh : Build the clang/llvm OpenMP compiler from the trunk master branch
#                    This is same as the AOMP build process but only 4 components needed
#                     llvm , lld, clang, and openmp
#

export AOMP_REPOS=$HOME/git/trunk
export AOMP=/tmp/$USER/install/trunk
#export BUILD_AOMP=/tmp/$USER/trunk
export AOMP_LLVM_REPO_BRANCH=master
export AOMP_LLD_REPO_BRANCH=master
export AOMP_CLANG_REPO_BRANCH=master
export AOMP_OPENMP_REPO_BRANCH=master

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

function build_trunk_component() {
   $HOME/git/aomp/$AOMP_REPO_NAME/bin/build_$COMPONENT.sh
   rc=$?
   if [ $rc != 0 ] ; then 
      echo " !!!  build_trunk.sh: BUILD FAILED FOR COMPONENT $COMPONENT !!!"
      exit $rc
   fi  
   $HOME/git/aomp/$AOMP_REPO_NAME/bin/build_$COMPONENT.sh install
   rc=$?
   if [ $rc != 0 ] ; then 
      echo " !!!  build_trunk.sh: INSTALL FAILED FOR COMPONENT $COMPONENT !!!"
      exit $rc
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
echo " =================  START build_trunk.sh ==================="   
echo 

components="llvm lld clang openmp"
for COMPONENT in $components ; do 
   echo 
   echo " =================  BUILDING COMPONENT $COMPONENT ==================="   
   echo 
   build_trunk_component
   date
   echo " =================  DONE INSTALLING COMPONENT $COMPONENT ==================="   
done


echo 
date
echo " =================  END build_trunk.sh ==================="   
echo 
exit 0
