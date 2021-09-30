#!/bin/bash
#
#  init_aomp_repos.sh:  Standalong script to initialize aomp repos 
#                       using google repo command.
#   Usage:
#      cd /tmp
#      wget  https://github.com/ROCm-Developer-Tools/aomp/raw/aomp-dev/init_aomp_repos.sh
#      . init_aomp_repos.sh

GITROCDEV="https://github.com/ROCm-Developer-Tools"
AOMP_VERSION=${AOMP_VERSION:-13.1}
AOMP_REPOS=${AOMP_REPOS:-${HOME}/git/aomp${AOMP_VERSION}}
repobindir=$AOMP_REPOS/.bin
manifest_file="manifests/aomp_$AOMP_VERSION.xml"
mkdir -p $repobindir

# Check Python before using repo command
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

if [ ! -f $repobindir/repo ] ; then
   echo "================  GETTING repo COMMAND FROM GOOGLE ================"
   mkdir -p $repobindir
   curl https://storage.googleapis.com/git-repo-downloads/repo > $repobindir/repo 2>/dev/null
   chmod a+x $repobindir/repo
fi

echo "================  STARTING INTERNAL REPO INIT ================"
cd $AOMP_REPOS
echo $repobindir/repo init -u $GITROCDEV/aomp -b aomp-dev -m $manifest_file
$repobindir/repo init -u $GITROCDEV/aomp -b aomp-dev -m $manifest_file
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
      echo "       This causes a conflict between the repo command and git clone"
      echo "       Try these commands to initialize the aomp_repos:"
      echo "       "
      echo " cd /tmp "
      echo " wget https://github.com/ROCm-Developer-Tools/aomp/raw/aomp-dev/init_aomp_repos.sh"
      echo " chmod 755 init_aomp_repos.sh"
      echo " ./init_aomp_repos.sh"
      exit 1
fi
echo "repo sync Done! Repositories can be found at $AOMP_REPOS"

echo "================  STARTING BRANCH CHECKOUT ================"
# Loop through synced projects and checkout branch revision specified in manifest.
echo "$repobindir/repo forall -pc 'git checkout \$REPO_RREV'"
$repobindir/repo forall -pc 'git checkout $REPO_RREV'
if [ $? != 0 ] ; then
   echo "$repobindir/repo forall checkout failed."
   exit 1
fi

# Loop through project groups thare are revlocked and checkout specific hash.
echo "$repobindir/repo forall -p -g revlocked -c 'git checkout \$REPO_UPSTREAM; git checkout \$REPO_RREV'"
$repobindir/repo forall -p -g revlocked -c 'git checkout $REPO_UPSTREAM; git checkout $REPO_RREV'
if [ $? != 0 ] ; then
   echo "$repobindir/repo forall revlocked checkout failed."
   exit 1
fi

# Finally run git pull for all unlocked projects.
echo $repobindir/repo forall -p -g unlocked -c \'git pull\'
$repobindir/repo forall -p -g unlocked -c 'git pull'
if [ $? != 0 ] ; then
   echo "$repobindir/repo forall git pull failed."
   exit 1
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

exit 0
