#!/bin/bash
#
#  aomp_internal_repo_sync:  get and/or update internal repos
# 
#  This script uses the file manifests/aomp-internal.xml from the aomp github.
#  This script requires you have internal access.

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
# --- end standard header ----A

function get_branch_name(){
   branch_name="UNKNOWN"
      [[ "$repodirname" == "aomp" ]] && branch_name=$AOMP_REPO_BRANCH
      [[ "$repodirname" == "llvm-project" ]] && branch_name=$AOMP_PROJECT_REPO_BRANCH
      [[ "$repodirname" == "aomp-extras" ]] && branch_name=$AOMP_EXTRAS_REPO_BRANCH
      [[ "$repodirname" == "flang" ]] && branch_name=$AOMP_FLANG_REPO_BRANCH
      [[ "$repodirname" == "roct-thunk-interface" ]] && branch_name=$AOMP_ROCT_REPO_BRANCH
      [[ "$repodirname" == "hsa-runtime" ]] && branch_name=$AOMP_HSA_RUNTIME_BRANCH_NAME
      [[ "$repodirname" == "rocminfo" ]] && branch_name=$AOMP_RINFO_REPO_BRANCH
      [[ "$repodirname" == "ROCgdb" ]] && branch_name=$AOMP_GDB_REPO_BRANCH
      [[ "$repodirname" == "ROCdbgapi" ]] && branch_name=$AOMP_DBGAPI_REPO_BRANCH
      [[ "$repodirname" == "rocm-device-libs" ]] && branch_name=$AOMP_LIBDEVICE_REPO_BRANCH
      [[ "$repodirname" == "rocprofiler" ]] && branch_name=$AOMP_PROF_REPO_BRANCH
      [[ "$repodirname" == "roctracer" ]] && branch_name=$AOMP_TRACE_REPO_BRANCH
      [[ "$repodirname" == "opencl-on-vdi" ]] && branch_name=$AOMP_OCL_REPO_BRANCH
      [[ "$repodirname" == "hip-on-vdi" ]] && branch_name=$AOMP_HIPVDI_REPO_BRANCH
      [[ "$repodirname" == "vdi" ]] && branch_name=$AOMP_VDI_REPO_BRANCH

      [[ "$repodirname" == "rocm-compilersupport" ]] && branch_name=$AOMP_COMGR_REPO_SHA
}

function list_repo(){
for repodirname in `ls $AOMP_REPOS` ; do
   if [[ "$repodirname" != "rocr-runtime"  && "$repodirname" != "build" ]] ; then
      fulldirname=$AOMP_REPOS/$repodirname
      get_branch_name
      cd $fulldirname
      abranch=`git branch | awk '/\*/ { print $2; }'`
      echo `git config --get remote.origin.url` "$repodirname  desired: " $branch_name" actual: " $abranch "  " `git log --numstat --format="%h" |head -1`
   fi
done
exit 0
}

LISTONLY=$1
if [ "$LISTONLY" == 'list' ]; then
list_repo
fi

ping -c 1 $AOMP_GIT_INTERNAL_IP 2> /dev/null >/dev/null
if [ $? != 0 ]; then
   echo ERROR: you must be internal AMD network to get internal repos
   exit 1
fi
if [[ "$AOMP_VERSION" != "13.1" ]] ; then
  echo "ERROR: Currently $0 only works with development of 13.1"
  echo "       You have AOMP_VERSION set to $AOMP_VERSION"
  echo "       To use $0 set AOMP_VERSION to \"13.1\""
  exit 1
fi
which python3  >/dev/null
if [ $? != 0 ] ; then 
   echo "ERROR:  No python found in path"
   echo "        Please install python3.6 or greater."
   exit 1
fi
pyver=`python3 --version | cut -d" " -f2 | cut -d"." -f1`
if [ "$pyver" != "3" ] ; then
   echo "ERROR:  the 'python' executeable must be python3"
   echo "        Try alias python to python3"
   exit 1
fi

repobindir=$AOMP_REPOS/.bin
mkdir -p $repobindir
if [ ! -f $repobindir/repo ] ; then

   echo "================  GETTING repo COMMAND FROM GOOGLE ================"
   mkdir -p $repobindir
   curl https://storage.googleapis.com/git-repo-downloads/repo > $repobindir/repo 2>/dev/null
   chmod a+x $repobindir/repo

   echo "==============  INSTALLING new python3 packages needed by internal roctracer repo ============"
   python3 -m pip install CppHeaderParser argparse wheel lit
   echo

fi

   echo "================  STARTING INTERNAL REPO INIT ================"
   cd $AOMP_REPOS
   $repobindir/repo init -u $GITROCDEV/$AOMP_REPO_NAME -b aomp-dev -m manifests/aomp-internal.xml
   if [ $? != 0 ] ; then
      echo "ERROR:  $repobindir/repo init failed"
      exit 1
   fi

   echo "================  STARTING INTERNAL REPO SYNC ================"

   echo $repobindir/repo sync
   $repobindir/repo sync
   if [ $? != 0 ] ; then
      echo
      echo "ERROR: $repobindir/repo sync failed. This could be because you used"
      echo "       git clone to fetch the original aomp master repository into "
      echo "       directory $AOMP_REPOS"
      echo "       You MUST git clone into a different directory for the initial"
      echo "       copy of the aomp repository. Otherwise, the repo sync"
      echo "       command will not be able to correctly fetch aomp repo.  Try:"
      echo "       "
      echo "       git clone http://github.com/rocm-developer-tools/aomp /tmp/aomp_temp"
      echo "       /tmp/aomp_temp/bin/aomp_internal_repo_sync.sh"
      exit 1
   fi
   echo
   echo "Done repo sync"
   echo "The internal repositories can be found at $AOMP_REPOS"


echo
echo "================  Get REPOS on correct internal branches ================"
for repodirname in `ls $AOMP_REPOS` ; do
   # The build and rocr-runtime are not git repos, skip them.
   if [[ "$repodirname" != "rocr-runtime"  && "$repodirname" != "build" ]] ; then 
      echo
      get_branch_name
      echo "cd $AOMP_REPOS/$repodirname  "
      cd $AOMP_REPOS/$repodirname
      echo "git checkout $branch_name  "
      git checkout $branch_name
      if [[ "$repodirname" != "rocm-compilersupport" ]] ; then
        echo "git pull"
        git pull
      fi
      cd ..
   fi
done

# build_aomp.sh expects a repo at the direoctory for rocr-runtime
# Link in the open source hsa-runtime as "src" directory
if [ ! -L $AOMP_REPOS/rocr-runtime/src ] ; then
   echo "Fixing rocr-runtime with correct link to hsa-runtime/opensrc/hsa-runtime src"
   mkdir -p $AOMP_REPOS/rocr-runtime
   cd $AOMP_REPOS/rocr-runtime
   echo ln -sf $AOMP_REPOS/hsa-runtime/opensrc/hsa-runtime src
   ln -sf $AOMP_REPOS/hsa-runtime/opensrc/hsa-runtime src
fi

cd $AOMP_REPOS/hsa-runtime
git checkout $AOMP_HSA_RUNTIME_BRANCH_NAME
git pull

echo
echo "========== $0 IS COMPLETE in $AOMP_REPOS=========="

exit 0
