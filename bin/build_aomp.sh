#!/bin/bash
# 
#   build_aomp.sh : Build all AOMP components 
#
if [ "$1" == "clean" ]; then
  echo "Exiting build, clean argument unknown. Try '--clean'."
  exit 1
fi

echo "ls $INSTALL_PREFIX/llvm/bin"
ls $INSTALL_PREFIX/llvm/bin

# Force clean because --clean is not being called correctly
if [ "$AOMP_STANDALONE_BUILD" == 0 ] ; then
  echo "ls $OUT_DIR/build/"
  ls $OUT_DIR/build/

  echo "Clean install directory:"
  echo "rm -rf $INSTALL_PREFIX/openmp-extras/*"
  rm -rf $INSTALL_PREFIX/openmp-extras/*

  echo "Clean build directory:"
  echo "rm -rf $OUT_DIR/build/openmp-extras/*"
  rm -rf "$OUT_DIR/build/openmp-extras/*"

  echo "ls $INSTALL_PREFIX/openmp-extras"
  ls $INSTALL_PREFIX/openmp-extras

  echo "ls $OUT_DIR/build/"
  ls $OUT_DIR/build/

  if [ -d $OUT_DIR/build/openmp-extras ]; then
    echo "ls $OUT_DIR/build/openmp-extras"
    ls "$OUT_DIR/build/openmp-extras"
  else
    echo "$OUT_DIR/build/openmp-extras has been removed"
  fi
fi

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
   echo "Number of Arguments: $#"
   if [ $# -eq 0 ] || [ "$SANITIZER" == "1" ]; then
       echo "Installing $@"
       $AOMP_REPOS/$AOMP_REPO_NAME/bin/build_$COMPONENT.sh install
       echo ""
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
if [ -n "$AOMP_JENKINS_BUILD_LIST" ] ; then
   components=$AOMP_JENKINS_BUILD_LIST
   if [ "$SANITIZER" == 1 ] && [ -f $AOMP/bin/flang-legacy ] ; then
     components="extras openmp offload pgmath flang flang_runtime"
   else
     components="extras openmp offload flang-legacy pgmath flang flang_runtime"
   fi
else
   if [ "$AOMP_STANDALONE_BUILD" == 1 ] ; then
      # There is no good external repo for the opencl runtime but we only need the headers for build_vdi.sh
      # So build_ocl.sh is currently not called.
      components="roct rocr project libdevice extras openmp pgmath flang flang_runtime comgr rocminfo vdi hipvdi "
   else
      # With AOMP 11, ROCM integrated build will not need roct rocr libdevice comgr and rocminfo
      #               In the future, when ROCm build vdi and hipvdi we can remove them
      components="project extras openmp pgmath flang flang_runtime vdi hipvdi"
   fi
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
if [ "$AOMP_STANDALONE_BUILD" -eq 0 ]; then
  cd $BUILD_DIR/build
  legacy_version=`ls flang-legacy`
  legacy_install_manifest=$legacy_version/install_manifest.txt
  if [ "$SANITIZER" == 1 ]; then
    install_manifest_orig=asan/install_manifest.txt
  else
    install_manifest_orig=install_manifest.txt
  fi

  # Clean file log
  rm -f $BUILD_DIR/build/installed_files.txt

  for directory in ./*/; do
    pushd $directory > /dev/null
    if [[ "$directory" =~ "flang-legacy" ]]; then
      install_manifest=$legacy_install_manifest
    else
      install_manifest=$install_manifest_orig
    fi
    if [ -f "$install_manifest" ]; then
      cat $install_manifest  >> $BUILD_DIR/build/installed_files.txt
      echo "" >> $BUILD_DIR/build/installed_files.txt
    fi
    popd > /dev/null
  done
fi

echo "ls $INSTALL_PREFIX/openmp-extras:"
ls $INSTALL_PREFIX/openmp-extras
echo

echo "ls $INSTALL_PREFIX/openmp-extras/bin:"
ls $INSTALL_PREFIX/openmp-extras/bin
echo

echo "ls $INSTALL_PREFIX/openmp-extras/rocm-bin:"
ls $INSTALL_PREFIX/openmp-extras/rocm-bin
echo

#PATH=$INSTALL_PREFIX/llvm/bin:$PATH $AOMP_REPOS/$AOMP_REPO_NAME/bin/bashtest
#PATH=$INSTALL_PREFIX/llvm/bin:$PATH $AOMP_REPOS/$AOMP_REPO_NAME/bin/bashtestf90
exit 0
