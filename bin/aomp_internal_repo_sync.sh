#!/bin/bash
#
#  aomp_internal_repo_sync:  get and/or update internal repos
# 
#  This script uses the file manifests/aomp-internal.xml from the aomp github.
#  This script requires you have internal access to AMD vpn.

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
echo
echo "==============  INSTALLING new python3 packages needed by internal roctracer repo ============"
python3 -m pip install CppHeaderParser argparse wheel lit
echo

repobindir=$AOMP_REPOS/.bin

mkdir -p $repobindir
curl https://storage.googleapis.com/git-repo-downloads/repo > $repobindir/repo 2>/dev/null
chmod a+x $repobindir/repo

echo "================  STARTING INTERNAL REPO INIT ================"
cd $AOMP_REPOS
$repobindir/repo init -u $GITROCDEV/$AOMP_REPO_NAME -b aomp-dev -m manifests/aomp-internal.xml
if [ $? != 0 ] ; then 
   echo "ERROR:  $repobindir/repo init failed"
   exit 1
fi
echo
echo "================  STARTING INTERNAL REPO SYNC ================"

echo $repobindir/repo sync
$repobindir/repo sync
if [ $? != 0 ] ; then 
   echo "ERROR:  $repobindir/repo sync failed"
   exit 1
fi
echo
echo "Done repo sync"
echo "The internal repositories can be found at $AOMP_REPOS"
echo

echo
echo "================  Get REPOS on correct internal branches ================"
for reponame in `ls $AOMP_REPOS` ; do 
   echo
   if [ $reponame == "rocr-runtime" ] ; then
      if [ -L $AOMP_REPOS/rocr-runtime/src ] ; then
         echo Already linked rocr-runtime to hsa-runtime/opensrc/hsa-runtime/src
      else
         echo "Fixing rocr-runtime with correct link to hsa-runtime/opensrc/hsa-runtime src"
         mkdir -p $AOMP_REPOS/rocr-runtime
         cd $AOMP_REPOS/rocr-runtime
         echo ln -sf $AOMP_REPOS/hsa-runtime/opensrc/hsa-runtime src
         ln -sf $AOMP_REPOS/hsa-runtime/opensrc/hsa-runtime src
      fi
   else
      if [[ "$reponame" == "aomp" || "$reponame" == "aomp-extras" || "$reponame" == "llvm-project" || "$reponame" == "flang" ]] ; then
         branch_name="aomp-dev"
      elif [[ "$reponame" == "roct-thunk-interface"  ||  "$reponame" == "rocminfo" ||  "$reponame" == "ROCgdb" ||  "$reponame" == "ROCdbgapi" ]] ; then
         branch_name="amd-staging"
      elif [[ "$reponame" == "rocm-device-libs" || "$reponame" == "rocm-compilersupport" ]] ; then
         branch_name="amd-stg-open"
      else
         branch_name="amd-master"
      fi
      echo "cd $AOMP_REPOS/$reponame ; git checkout $branch_name"
      cd $AOMP_REPOS/$reponame
      git checkout $branch_name
      git pull
      cd ..
   fi
done

echo
echo "========== $0 IS COMPLETE in $AOMP_REPOS=========="

exit 0
