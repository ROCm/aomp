#!/bin/bash
# 
#   build_f18.sh : Build all F18 components.
#      For now, only build project component from firdev-llvm-project repo.
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
. $thisdir/f18_common_vars
# --- end standard header ----

function build_aomp_component() {
   [ -f /opt/rh/devtoolset-7/enable ] &&  . /opt/rh/devtoolset-7/enable
   $F18_REPOS/$AOMP_REPO_NAME/bin/build_$COMPONENT.sh "$@"
   rc=$?
   if [ $rc != 0 ] ; then 
      echo " !!!  build_aomp.sh: BUILD FAILED FOR COMPONENT $COMPONENT !!!"
      exit $rc
   fi  
   if [ $# -eq 0 ] ; then
       $F18_REPOS/$AOMP_REPO_NAME/bin/build_$COMPONENT.sh install
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

# Test update access to F18_INSTALL_DIR
# This should be done early to ensure sudo (if set) does not prompt for password later
$TOPSUDO mkdir -p $F18_INSTALL_DIR
if [ $? != 0 ] ; then
   echo "ERROR: $TOPSUDO mkdir failed, No update access to $F18_INSTALL_DIR"
   exit 1
fi
$TOPSUDO touch $F18_INSTALL_DIR/testfile
if [ $? != 0 ] ; then
   echo "ERROR: $TOPSUDO touch failed, No update access to $F18_INSTALL_DIR"
   exit 1
fi
$TOPSUDO rm $F18_INSTALL_DIR/testfile

#Check for gawk on Ubuntu, which is needed for the flang build.
GAWK=$(gawk --version | grep "^GNU Awk")
OS=$(cat /etc/os-release | grep "^NAME=")

if [[ -z $GAWK ]] && [[ "$OS" == *"Ubuntu"* ]] ; then
   echo
   echo "Build Error: gawk was not found and is required for building flang! Please run 'sudo apt-get install gawk' and run build_f18.sh again."
   echo
   exit 1
fi

echo 
date
echo " =================  START build_f18.sh ==================="   
echo 
   if [ "$F18_STANDALONE_BUILD" == 1 ] ; then
      # There is no good external repo for the opencl runtime but we only need the headers for build_vdi.sh
      # So build_ocl.sh is currently not called.
      #components="roct rocr project libdevice extras openmp pgmath flang flang_runtime comgr rocminfo vdi hipvdi ocl "
      components="project"
   else
      # With AOMP 11, ROCM integrated build will not need roct rocr libdevice comgr and rocminfo
      #               In the future, when ROCm build vdi and hipvdi we can remove them
      components="project"
   fi
echo "COMPONENTS:$components"

#Partial build options. Check if argument was given.
if [ -n "$1" ] ; then
  found=0
#Start build from given component (./build_f18.sh continue openmp)
  if [ "$1" == 'continue' ] ; then
    for COMPONENT in $components ; do
      if [ $COMPONENT == "$2" ] ; then
        found=1
      fi
      if [[ $found -eq 1 ]] ; then
        list+="$COMPONENT "
      fi
    done
    components=$list
    if [[ $found -eq 0 ]] ; then
      echo "$2 was not found in the build list!!!"
    fi
    #Remove arguments so they are not passed to build_aomp_component
    set --

  #Select which components to build(./build_f18.sh select libdevice extras)
  elif [ "$1" == 'select' ] ; then
    for ARGUMENT in $@ ; do
      if [ $ARGUMENT != "$1" ] ; then
        list+="$ARGUMENT "
      fi
    done
    components=$list
    #Remove arguments so they are not passed to build_aomp_component
    set --
  fi
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
#$F18_REPOS/$AOMP_REPO_NAME/bin/build_fixups.sh
echo 
date
echo " =================  END build_f18.sh ==================="   
echo 
exit 0
