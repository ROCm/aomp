#!/bin/bash
# 
#   build_rocmlibs.sh : Build and install ROCm libraries as AOMP components 
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/../aomp_common_vars
# --- end standard header ----

_total_file_count=0
_total_file_size=0
_total_secs=0
_total_start_date=`date`

function build_rocmlib_component() {
   _stats_dir=$AOMP_INSTALL_DIR/.aomp_component_stats
   mkdir -p $_stats_dir
   touch $_stats_dir/.${COMPONENT}.ts
   start_date=`date`
   start_secs=`date +%s`

   if [ "$COMPONENT" == "prereq" ] ; then 
      $AOMP_REPOS/$AOMP_REPO_NAME/bin/build_prereq.sh "$@"
   elif [ "$COMPONENT" == "rocm-cmake" ] ; then 
      $AOMP_REPOS/$AOMP_REPO_NAME/bin/build_rocm-cmake.sh "$@"
   else
      $AOMP_REPOS/$AOMP_REPO_NAME/bin/rocmlibs/build_$COMPONENT.sh "$@"
   fi
   rc=$?
   if [ $rc != 0 ] ; then 
      echo " !!!  build_rocmlibs.sh: BUILD FAILED FOR COMPONENT $COMPONENT !!!"
      exit $rc
   fi  
   if [ $# -eq 0 ] ; then
       if [ "$COMPONENT" == "prereq" ] ; then 
          $AOMP_REPOS/$AOMP_REPO_NAME/bin/build_prereq.sh install
       elif [ "$COMPONENT" == "rocm-cmake" ] ; then 
          $AOMP_REPOS/$AOMP_REPO_NAME/bin/build_rocm-cmake.sh install
       else
          $AOMP_REPOS/$AOMP_REPO_NAME/bin/rocmlibs/build_$COMPONENT.sh install
       fi
       rc=$?
       if [ $rc != 0 ] ; then 
           echo " !!!  $0 install FAILED FOR COMPONENT $COMPONENT !!!"
           exit $rc
       fi
       # gather stats on artifacts installed with this component build
       end_date=`date`
       end_secs=`date +%s`
       _component_secs=$(( $end_secs - $start_secs ))
       find $AOMP_INSTALL_DIR -type f -newer $_stats_dir/.${COMPONENT}.ts | xargs wc -c >$_stats_dir/$COMPONENT.files
       echo "COMPONENT $COMPONENT START : $start_date " >$_stats_dir/$COMPONENT.stats
       echo "COMPONENT $COMPONENT END   : $end_date" >>$_stats_dir/$COMPONENT.stats
       echo "COMPONENT $COMPONENT TIME  : $_component_secs seconds" >> $_stats_dir/$COMPONENT.stats
       file_count=`wc -l $_stats_dir/$COMPONENT.files | cut -d" " -f1`
       file_count=$(( file_count - 1 ))
       echo "COMPONENT $COMPONENT FILES : $file_count " >> $_stats_dir/$COMPONENT.stats
       new_bytes=`grep " total" $_stats_dir/$COMPONENT.files | cut -d" " -f1 | awk '{sum += $1} END {print sum}'`
       k_bytes=$(( new_bytes / 1024 ))
       m_bytes=$(( k_bytes / 1024 ))
       echo "COMPONENT $COMPONENT SIZE  : $k_bytes KB  $m_bytes MB " >> $_stats_dir/$COMPONENT.stats
 
       # Keep running totals for status on all rocmlibs
       _total_file_size=$(( _total_file_size + m_bytes ))
       _total_file_count=$(( _total_file_count + $file_count ))
       _total_secs=$(( _total_secs + $_component_secs ))
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
echo " =================  START build_rocmlibs.sh ==================="   
echo 
components="prereq rocm-cmake"

if [ "$AOMP_STANDALONE_BUILD" == 1 ] ; then
  # This ordered build is important when starting from scratch
  components="$components rocblas rocprim rocsparse rocsolver hipblas"
else
  echo "ERROR: Cannot run $0 with AOMP_STANDALONE_BUILD=$AOMP_STANDALONE_BUILD"
  echo "       Please set $AOMP_STANDALONE_BUILD=1"
  exit 1
fi
echo "COMPONENTS:$components"

#Partial build options. Check if argument was given.
if [ -n "$1" ] ; then
  found=0
#Start build from given component (./build_rocmlibs.sh continue rocblas)
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
    #Remove arguments so they are not passed to build_rocmlib_component
    set --

  #Select which components to build(./build_rocmlibs.sh select rocblas hipblas )
  elif [ "$1" == 'select' ] ; then
    for ARGUMENT in $@ ; do
      if [ $ARGUMENT != "$1" ] ; then
        list+="$ARGUMENT "
      fi
    done
    components=$list
    #Remove arguments so they are not passed to build_rocmlib_component
    set --
  fi
fi
echo "components: $components"

for COMPONENT in $components ; do 
   echo 
   echo " =================  BUILDING COMPONENT $COMPONENT ==================="   
   echo 
   build_rocmlib_component "$@"
   date
   echo " =================  DONE INSTALLING COMPONENT $COMPONENT ==================="   
done

_total_end_date=`date`
echo "rocmlibs START : $_total_start_date "    >> $_stats_dir/rocmlibs.stats
echo "rocmlibs END   : $_total_end_date"       >> $_stats_dir/rocmlibs.stats
echo "rocmlibs TIME  : $_total_secs seconds"   >> $_stats_dir/rocmlibs.stats
echo "rocmlibs FILES : $_total_file_count "    >> $_stats_dir/rocmlibs.stats
echo "rocmlibs SIZE  : $_total_file_size MB "  >> $_stats_dir/rocmlibs.stats

date
echo " =================  END build_rocmlibs.sh ==================="   
echo 

exit 0
