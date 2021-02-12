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
if [ "$1" == "-undo" ] ; then
   if [ ! -d $AOMP_REPOS/INTERNAL ] ; then
      echo "ERROR: No $AOMP_REPOS/INTERNAL directory needed for -undo"
      exit 1
   fi
   if [ ! -d $AOMP_REPOS/EXTERNAL ] ; then
      echo "ERROR: No $AOMP_REPOS/EXTERNAL directory needed for -undo"
      exit 1
   fi
   echo
   echo "=====  UNDOING INTERNAL LINKS MADE BY $0  ====="
   for reponame in `ls $AOMP_REPOS/INTERNAL` ; do
      echo
      if [ -L $AOMP_REPOS/$reponame ] ; then
         echo Found link to INTERNAL/$reponame
	 if [ ! -d $AOMP_REPOS/EXTERNAL/$reponame ] ; then
            echo "ERROR: No $AOMP_REPOS/EXTERNAL/$reponame directory needed for -undo"
	    exit 1
         fi
	 echo rm $AOMP_REPOS/$reponame
	 rm $AOMP_REPOS/$reponame
	 echo mv $AOMP_REPOS/EXTERNAL/$reponame $AOMP_REPOS/$reponame
	 mv $AOMP_REPOS/EXTERNAL/$reponame $AOMP_REPOS/$reponame
      else
         if [ -d $AOMP_REPOS/$reponame ] ; then
            echo Directory already found at $AOMP_REPOS/$reponame , SKIPPING
         else
            if [ $reponame == "hsa-runtime" ] ; then
               if [ -L $AOMP_REPOS/rocr-runtime/src ] ; then
                  echo Link found from rocr-runtime to INTERNAL/hsa-runtime/opensrc/hsa-runtime/src
	          if [ ! -d $AOMP_REPOS/EXTERNAL/rocr-runtime ] ; then
		     echo "ERROR: Missing $AOMP_REPOSE/EXTERNAL/rocr-runtime to undo"
	          else
	             echo rm $AOMP_REPOS/rocr-runtime/src
	             rm $AOMP_REPOS/rocr-runtime/src
		     echo rm -rf $AOMP_REPOS/rocr-runtime
		     rm -rf $AOMP_REPOS/rocr-runtime
	             echo mv $AOMP_REPOS/EXTERNAL/rocr-runtime $AOMP_REPOS/rocr-runtime
	             mv $AOMP_REPOS/EXTERNAL/rocr-runtime $AOMP_REPOS/rocr-runtime
		  fi
	       else
	          echo No link found at $AOMP_REPOS/rocr-runtime/src , SKIPPING
               fi
            else
	       echo No link or directory found at $AOMP_REPOS/$reponame , SKIPPING
            fi
         fi
      fi
   done
   if [ -d $AOMP_REPOS/EXTERNAL ] ; then
      echo rmdir $AOMP_REPOS/EXTERNAL
      rmdir $AOMP_REPOS/EXTERNAL
      if [ $? != 0 ] ; then
         echo "WARNING: Could not rmdir $AOMP_REPOS/EXTERNAL , it should be empty after -undo"
      fi
   fi
   echo "===== $0 -undo IS COMPLETE  ====="
   ls -l $AOMP_REPOS
   exit 0
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
echo "=========  REPLACING REPOS WITH LINKS TO INTERNAL repos =========="

mkdir -p $AOMP_REPOS/EXTERNAL
for reponame in `ls $repodir` ; do 
   echo
   echo $reponame
   if [ -L $AOMP_REPOS/$reponame ] ; then 
      echo Already linked to INTERNAL
   else
      if [ -d $AOMP_REPOS/$reponame ] ; then 
         echo Directory found $AOMP_REPOS/$reponame ,moving to EXTERNAL
         mv $AOMP_REPOS/$reponame $AOMP_REPOS/EXTERNAL/.
         echo Linking to $repodir/$reponame
         echo ln -sf $repodir/$reponame $AOMP_REPOS/$reponame
         ln -sf $repodir/$reponame $AOMP_REPOS/$reponame 
      else
         if [ $reponame == "hsa-runtime" ] ; then 
            if [ -L $AOMP_REPOS/rocr-runtime/src ] ; then
               echo Already linked rocr-runtime to INTERNAL hsa-runtime/opensrc/hsa-runtime/src
            else
	       echo "Fixing rocr-runtime with correct link to hsa-runtime/opensrc/hsa-runtime src"
               mv $AOMP_REPOS/rocr-runtime $AOMP_REPOS/EXTERNAL/.
	       mkdir -p $AOMP_REPOS/rocr-runtime
	       cd $AOMP_REPOS/rocr-runtime
	       echo ln -sf $repodir/hsa-runtime/opensrc/hsa-runtime src
	       ln -sf $repodir/hsa-runtime/opensrc/hsa-runtime src
            fi
         else 
	    echo "WARNING: Don't know what to do with INTERNAL/$reponame"
	    echo "         No corresponding repo name or link in $AOMP_REPOS"
	    echo "         It Could be unnecessary repo or directory in manifest"
         fi
      fi
   fi
done

echo
echo "==============  INSTALLING new python3 packages needed by internal roctracer repo ============"
python3 -m pip install CppHeaderParser argparse wheel lit
echo
echo "========== $0 IS COMPLETE =========="
echo ls -l $AOMP_REPOS
ls -l $AOMP_REPOS
exit 0
