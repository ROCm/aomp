#!/bin/bash
#
#  aomp_internal_repo_sync:  get and/or update internal repos
# 
# --- Start standard header ----
if [ "$1" != "repocmd" ]; then

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

manifest_file="manifests/aomp_$AOMP_VERSION.xml"
local_manifest_file="$thisdir/../$manifest_file"
repobindir=$AOMP_REPOS/.bin

if [ "$1" == "list" ] ; then
  $repobindir/repo forall -g unlocked -c $thisdir/aomp_internal_repo_sync.sh repocmd list
  $repobindir/repo forall -g revlocked -c $thisdir/aomp_internal_repo_sync.sh repocmd list
  exit
fi

ping -c 1 $AOMP_GIT_INTERNAL_IP 2> /dev/null >/dev/null
if [ $? != 0 ]; then
   echo ERROR: you must be internal AMD network to get internal repos
   exit 1
fi
if [[ "$AOMP_VERSION" != "13.1" ]] && [[ $AOMP_MAJOR_VERSION -lt 14 ]] ; then
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
   $repobindir/repo init -u $GITROCDEV/$AOMP_REPO_NAME -b aomp-dev -m $manifest_file
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

echo "================  STARTING BRANCH CHECKOUT ================"
# Loop through synced projects and checkout branch revision specified in manifest.
echo "$repobindir/repo forall -pc 'git checkout \$REPO_RREV'"
$repobindir/repo forall -pc 'git checkout $REPO_RREV'
if [ $? != 0 ] ; then
   echo "WARNING: $repobindir/repo forall checkout failed."
   # exit 1
fi

# Loop through project groups thare are revlocked and checkout specific hash.
$repobindir/repo forall -p -g revlocked -c $thisdir/aomp_internal_repo_sync.sh repocmd gitcheckout
$repobindir/repo forall -p -g unlocked -c $thisdir/aomp_internal_repo_sync.sh repocmd gitcheckout
if [ $? != 0 ] ; then
   echo "WARNING: $repobindir/repo forall revlocked checkout failed."
   # exit 1
fi

echo "================  STARTING BRANCH PULL ================"
# Finally run git pull for all unlocked projects.
echo $repobindir/repo forall -p -g unlocked -c $thisdir/aomp_internal_repo_sync.sh repocmd gitpull
$repobindir/repo forall -p -g unlocked -c $thisdir/aomp_internal_repo_sync.sh repocmd gitpull
if [ $? != 0 ] ; then
   echo "WARNING: $repobindir/repo forall git pull failed."
   #exit 1
fi

# build_rocr.sh expects directory rocr-runtime which is a subdir of hsa-runtime
# Link in the open source hsa-runtime as "src" directory
if [ -d $AOMP_REPOS/hsa-runtime ] ; then
   if [ ! -L $AOMP_REPOS/rocr-runtime/src ] ; then
      echo "Fixing rocr-runtime with correct link to hsa-runtime/opensrc/hsa-runtime src"
      mkdir -p $AOMP_REPOS/rocr-runtime
      cd $AOMP_REPOS/rocr-runtime
      echo ln -sf $AOMP_REPOS/hsa-runtime/opensrc/hsa-runtime src
      ln -sf $AOMP_REPOS/hsa-runtime/opensrc/hsa-runtime src
   fi
fi

echo
echo "========== $0 IS COMPLETE in $AOMP_REPOS =========="

else

function list_repo(){
   rc=0
   repodirname=$REPO_PATH
   thisbranch="${REPO_REMOTE}/${REPO_RREV:0:12}"
      HASH=`git log -1 --numstat --format="%h" | head -1`
      flag=""
      is_hash=0
      branch_name=${REPO_RREV}
      # get the actual branch
      abranch=`git branch | awk '/\*/ { print $2; }'`
      if [ "$abranch" == "(no" ] ; then
         is_hash=1
         abranch=`git branch | awk '/\*/ { print $5; }' | cut -d")" -f1`
         if [ "$abranch" != "$HASH" ] ; then
	    flag="!HASH!	"
	    rc=1
	 fi
      fi
      url=`git config --get remote.origin.url`
      if [ "$url" == "" ] ; then
         url=`git config --get remote.roctools.url`
      fi
      if [ "$url" == "" ] ; then
         url=`git config --get remote.gerritgit.url`
      fi
      if [ "$branch_name" != "$abranch" ] && [ $is_hash == 0 ] ; then
         flag="!BRANCH!	"
	 rc=1
      fi
      echo "$flag$repodirname  url:$url  desired:$branch_name  actual:$abranch  hash:$HASH"
}

  # first arg is not repocmd
  if [[ "$2" == "gitpull" ]] ; then
    echo calling git pull in directory $PWD
    git pull
  elif [[ "$2" == "gitcheckout" ]] ; then
    echo git checkout $REPO_RREV
    git checkout $REPO_RREV
  elif [[ "$2" == "list" ]] ; then
    list_repo
  else
    echo "ERROR: Bad subcommand for repocmd cmd:$2"
    exit 1
  fi

fi

exit 0
