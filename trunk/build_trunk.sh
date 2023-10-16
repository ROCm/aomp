#!/bin/bash
# 
#   build_trunk.sh : Build all trunk components.
#
# --- Start standard header ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/trunk_common_vars
# --- end standard header ----

function build_trunk_component() {
   [ -f /opt/rh/devtoolset-7/enable ] &&  . /opt/rh/devtoolset-7/enable
   $TRUNK_REPOS/aomp/trunk/build_$COMPONENT.sh "$@"
   rc=$?
   if [ $rc != 0 ] ; then 
      echo " !!!  build_trunk.sh: BUILD FAILED FOR COMPONENT $COMPONENT !!!"
      exit $rc
   fi  
   if [ $# -eq 0 ] ; then
       $TRUNK_REPOS/aomp/trunk/build_$COMPONENT.sh install
       rc=$?
       if [ $rc != 0 ] ; then 
           echo " !!!  build_trunk.sh: INSTALL FAILED FOR COMPONENT $COMPONENT !!!"
           exit $rc
       fi
   fi
}

# Test update access to TRUNK_INSTALL_DIR
# This should be done early to ensure sudo (if set) does not prompt for password later
mkdir -p $TRUNK_INSTALL_DIR
if [ $? != 0 ] ; then
   echo "ERROR: mkdir failed, No update access to $TRUNK_INSTALL_DIR"
   exit 1
fi
touch $TRUNK_INSTALL_DIR/testfile
if [ $? != 0 ] ; then
   echo "ERROR: touch failed, No update access to $TRUNK_INSTALL_DIR"
   exit 1
fi
rm $TRUNK_INSTALL_DIR/testfile

#Check for gawk on Ubuntu, which is needed for building flang in build_project.sh.
GAWK=$(gawk --version | grep "^GNU Awk")
OS=$(cat /etc/os-release | grep "^NAME=")

if [[ -z $GAWK ]] && [[ "$OS" == *"Ubuntu"* ]] ; then
   echo
   echo "Build Error: gawk was not found and is required for building flang! Please run 'sudo apt-get install gawk' and run build_trunk.sh again."
   echo
   exit 1
fi

echo 
date
echo " =================  START build_trunk.sh ==================="   
echo 
# Ensure the prereq components in $HOME/local are up to date.
# By default build_prereq.sh will build cmake rocmsmilib and hwloc
# For trunk, we only need the current cmake.
export PREREQUISITE_COMPONENTS=${PREREQUISITE_COMPONENTS:-cmake}
components="$TRUNK_COMPONENT_LIST"
echo "COMPONENTS:$components"

#Partial build options. Check if argument was given.
if [ -n "$1" ] ; then
  found=0
#Start build from given component (./build_trunk.sh continue openmp)
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
    #Remove arguments so they are not passed to build_trunk_component
    set --

  #Select which components to build(./build_trunk.sh select libdevice extras)
  elif [ "$1" == 'select' ] ; then
    for ARGUMENT in $@ ; do
      if [ $ARGUMENT != "$1" ] ; then
        list+="$ARGUMENT "
      fi
    done
    components=$list
    #Remove arguments so they are not passed to build_trunk_component
    set --
  fi
fi

for COMPONENT in $components ; do 
   echo 
   echo " =================  BUILDING COMPONENT $COMPONENT ==================="   
   echo 
   build_trunk_component "$@"
   date
   echo " =================  DONE INSTALLING COMPONENT $COMPONENT ==================="   
done
echo 
date
echo " =================  END build_trunk.sh ==================="   
echo 
exit 0
