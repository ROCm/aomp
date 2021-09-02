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
   osversion=$(cat /etc/os-release | grep -e ^VERSION_ID)

   if [[ $osversion =~ '"7.' ]]; then
     echo "OS version 7 found `cat /etc/os-release`"
     [ -f /opt/rh/devtoolset-7/enable ] &&  . /opt/rh/devtoolset-7/enable
   elif [[ $osversion =~ '"8' ]]; then
     echo "OS version 8 found `cat /etc/os-release`"
     echo
     echo "Get updated gcc 8: export PATH=/usr/bin:\$PATH"
     export PATH=/usr/bin:$PATH
     gcc --version
   fi

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

#Check for gawk on Ubuntu, which is needed for the flang build.
GAWK=$(gawk --version | grep "^GNU Awk")
OS=$(cat /etc/os-release | grep "^NAME=")

if [[ -z $GAWK ]] && [[ "$OS" == *"Ubuntu"* ]] ; then
   echo
   echo "Build Error: gawk was not found and is required for building flang! Please run 'sudo apt-get install gawk' and run build_aomp.sh again."
   echo
   exit 1
fi

echo 
date
echo " =================  START build_aomp.sh ==================="   
echo 
if [ "$AOMP_STANDALONE_BUILD" == 1 ] ; then
  components="roct rocr project libdevice openmp extras comgr rocminfo"
  if [ $AOMP_MAJOR_VERSION -gt 13 ] ; then
     components="rocm-cmake $components"
  fi
  _hostarch=`uname -m`
  # The VDI (rocclr) architecture is very x86 centric so it will not build on ppc64. Without
  # rocclr, we have no HIP or OpenCL for ppc64 :-( However, rocr works for ppc64 so AOMP works.
  if [ "$_hostarch" == "x86_64" ] ; then
    # These components build on x86_64, so add them to components list
    components="$components pgmath flang flang_runtime"
    if [ "$AOMP_VERSION" == "13.1" ] || [ $AOMP_MAJOR_VERSION -gt 13 ] ; then
       components="$components hipamd "
    else
       components="$components vdi hipvdi ocl "
    fi
  fi

  # ROCdbgapi requires atleast g++ 7
  GPPVERS=`g++ --version | grep g++ | cut -d")" -f2 | cut -d"." -f1`
  if [ "$AOMP_BUILD_DEBUG" == "1" ] && [ "$GPPVERS" -ge 7 ]; then
    components="$components rocdbgapi rocgdb"
  fi
  # Do not add roctracer/rocprofiler for tarball install
  if [ "$TARBALL_INSTALL" != 1 ] && [ "$_hostarch" == "x86_64" ] ; then
    components="$components roctracer rocprofiler"
  fi
else
  # For ROCM build (AOMP_STANDALONE_BUILD=0) the components roct, rocr,
  # libdevice, comgr, rocminfo, vdi, hipvdi, ocl, rocdbgapi rocgdb,
  # roctracer, and rocprofiler should be found in ROCM in /opt/rocm.
  # The ROCM build only needs these components:
  components="project extras openmp pgmath flang flang_runtime"
fi
echo "COMPONENTS:$components"

#Partial build options. Check if argument was given.
if [ -n "$1" ] ; then
  found=0
#Start build from given component (./build_aomp.sh continue openmp)
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

  #Select which components to build(./build_aomp.sh select libdevice extras)
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
#$AOMP_REPOS/$AOMP_REPO_NAME/bin/build_fixups.sh
echo 
date
echo " =================  END build_aomp.sh ==================="   
echo 
exit 0
