#!/bin/bash
#
#  aomp_internal_repo_sync:  get and/or update internal repos, then link them correctly for build_aomp.sh.
#                            Internal Repos will be put in $AOMP_REPOS/INTERNAL.
#                            External repos are moved to $AOMP_REPOS/EXTERNAL before
#                            the symlinks are created. This script also downloads and installs
#                            the repo command from google. 
# 
#  This script uses the file manifests/aomp-internal.xml from the aomp github.
#  This script requires you have internal access to AMD vpn.
#  WARNING:  I don't have a utility to replace INTERNAL repos with EXTERNAL, That is,
#            I dont YET have a utility to undo the moving and linking that this script does.

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

# Put all internal repos in a subdirectory called INTERNAL
repodir=$AOMP_REPOS/INTERNAL
repobindir=$repodir/.bin

mkdir -p $repobindir
curl https://storage.googleapis.com/git-repo-downloads/repo > $repobindir/repo 2>/dev/null
chmod a+x $repobindir/repo

echo "================  STARTING INTERNAL REPO INIT ================"
cd $repodir
echo $repobindir/repo init -u $GITROCDEV/$AOMP_REPO_NAME -b amd-stg-openmp -m manifests/aomp-internal.xml
$repobindir/repo init -u $GITROCDEV/$AOMP_REPO_NAME -b amd-stg-openmp -m manifests/aomp-internal.xml
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
echo "The internal repositories can be found at $repodir"
echo 
echo "==============  STARTING REPLACING EXTERNAL WITH INTERNAL ==============="

mkdir -p $AOMP_REPOS/EXTERNAL
for reponame in `ls $repodir` ; do 
   echo
   echo $reponame
   if [ -L $AOMP_REPOS/$reponame ] ; then 
      echo already linked
   else
      if [ -d $AOMP_REPOS/$reponame ] ; then 
         echo Directory found $AOMP_REPOS/$reponame
         mv $AOMP_REPOS/$reponame $AOMP_REPOS/EXTERNAL/.
         ln -sf $repodir/$reponame $AOMP_REPOS/$reponame 
      else
         if [ $reponame == "hsa-runtime" ] ; then 
            if [ -L $AOMP_REPOS/rocr-runtime ] ; then 
                echo already linked
            else
	       echo "fixing hsa-runtime"
               mv $AOMP_REPOS/rocr-runtime $AOMP_REPOS/EXTERNAL/.
	       mkdir -p $AOMP_REPOS/rocr-runtime
	       echo ln -sf $repodir/hsa-runtime/opensrc/hsa-runtime $AOMP_REPOS/rocr-runtime/src
	       ln -sf $repodir/hsa-runtime/opensrc/hsa-runtime $AOMP_REPOS/rocr-runtime/src
            fi
         else 
	    echo "ERROR: dont know what to do with INTERNAL/$reponame"
         fi
      fi
   fi
done


