#!/bin/bash
# 
#   build_aomp.sh : Build all AOMP components 
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
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

   _stats_dir=$AOMP_INSTALL_DIR/.aomp_component_stats
   mkdir -p $_stats_dir
   touch $_stats_dir/.${COMPONENT}.ts
   start_date=`date`
   start_secs=`date +%s`

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
       # gather stats on artifacts installed with this component build
       end_date=`date`
       end_secs=`date +%s`
       find $AOMP_INSTALL_DIR -type f -newer $_stats_dir/.${COMPONENT}.ts | xargs wc -c >$_stats_dir/$COMPONENT.files
       echo "COMPONENT $COMPONENT START : $start_date " >$_stats_dir/$COMPONENT.stats
       echo "COMPONENT $COMPONENT END   : $end_date" >>$_stats_dir/$COMPONENT.stats
       echo "COMPONENT $COMPONENT TIME  : $(( $end_secs - $start_secs )) seconds" >> $_stats_dir/$COMPONENT.stats
       file_count=`wc -l $_stats_dir/$COMPONENT.files | cut -d" " -f1`
       file_count=$(( file_count -1 ))
       echo "COMPONENT $COMPONENT FILES : $file_count " >> $_stats_dir/$COMPONENT.stats
       new_bytes=`grep " total" $_stats_dir/$COMPONENT.files | cut -d" " -f1 | awk '{sum += $1} END {print sum}'`
       k_bytes=$(( new_bytes / 1024 ))
       m_bytes=$(( k_bytes / 1024 ))
       echo "COMPONENT $COMPONENT SIZE  : $k_bytes KB  $m_bytes MB " >> $_stats_dir/$COMPONENT.stats
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

if [ "$DISABLE_LLVM_TESTS" == "1" ]; then
  export DO_TESTS="-DLLVM_INCLUDE_TESTS=OFF -DCLANG_INCLUDE_TESTS=OFF"
fi

echo 
date
echo " =================  START build_aomp.sh ==================="   
echo 
components="$AOMP_COMPONENT_LIST"

if [ "$AOMP_STANDALONE_BUILD" == 1 ] ; then
  components="$components roct rocr openmp offload extras comgr rocminfo rocm_smi_lib amdsmi"
  _hostarch=`uname -m`
  # The rocclr architecture is very x86 centric so it will not build on ppc64. Without
  # rocclr, we have no HIP or OpenCL for ppc64 :-( However, rocr works for ppc64 so AOMP works.
  if [ "$_hostarch" == "x86_64" ] ; then
    # These components build on x86_64, so add them to components list
    if [ "$AOMP_SKIP_FLANG" == 0 ] ; then
      components="$components flang-legacy pgmath flang flang_runtime"
    fi
    #components="$components hipfort"
    components="$components hipcc hipamd "
  fi

  # ROCdbgapi requires atleast g++ 7
  GPPVERS=`g++ --version | grep g++ | cut -d")" -f2 | cut -d"." -f1`
  if [ "$AOMP_BUILD_DEBUG" == "1" ] && [ "$GPPVERS" -ge 7 ]; then
    components="$components rocdbgapi rocgdb"
  fi
  # Do not add roctracer/rocprofiler for tarball install
  # Also, as of ROCm 5.3 roctracer and rocprofiler require a rocm installation
  # The preceeding AOMP installation is not sufficient to build them.
  if [ "$TARBALL_INSTALL" != 1 ] && [ "$_hostarch" == "x86_64" ] ; then
    components="$components roctracer rocprofiler"
  fi
else
  # For ROCM build (AOMP_STANDALONE_BUILD=0) the components roct, rocr,
  # libdevice, project, comgr, rocminfo, hipamd, rocdbgapi, rocgdb,
  # roctracer, rocprofiler, rocm_smi_lib, and amdsmi should be found
  # in ROCM in /opt/rocm.  The ROCM build only needs these components:
  components="extras openmp"
  if [ -f "$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload/CMakeLists.txt" ]; then
    components="$components offload"
  fi
  if [ "$SANITIZER" == 1 ] && [ -f $AOMP/bin/flang-legacy ] ; then
    components="$components pgmath flang flang_runtime"
  else
    components="$components flang-legacy pgmath flang flang_runtime"
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
echo "components: $components"

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
exit 0
