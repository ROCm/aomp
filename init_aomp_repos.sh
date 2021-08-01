#!/bin/bash
#
#  init_aomp_repos.sh:  Standalong script to initialize aomp repos 
#                       using google repo command.
#   Usage:
#      cd /tmp; wget https://github.com/ROCm-Developer-Tools/aomp/init_aomp_repos.sh
#      . init_aomp_repos.sh

# FIXME: In the future, use version specific manifest_files
manifest_file="manifests/aomp-internal.xml"
GITROCDEV="https://github.com/ROCm-Developer-Tools"
AOMP_VERSION=${AOMP_VERSION:-13.1}
AOMP_REPOS=${AOMP_REPOS:-${HOME}/git/aomp${AOMP_VERSION}}
repobindir=$AOMP_REPOS/.bin
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
      echo " cd /tmp ; wget $GITROCDEV/aomp/init_aomp_repos.sh"
      echo " chmod 755 init_aomp_repos.sh"
      echo " ./init_aomp_repos.sh"
      echo " $AOMP_REPOS/aomp/bin/clone_aomp.sh"
      exit 1
fi
echo "repo sync Done! Repositories can be found at $AOMP_REPOS"
exit 0
