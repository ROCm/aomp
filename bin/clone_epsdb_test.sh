#!/bin/bash
#
#  clone_aomp.sh:  Clone the repositories needed to build the aomp compiler.  
#                  Currently AOMP needs 14 repositories.
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

function clone_or_pull(){
repodirname=$AOMP_REPOS_TEST/$reponame
echo
if [ -d $repodirname  ] ; then 
   echo "--- Pulling updates to existing dir $repodirname ----"
   echo "    We assume this came from an earlier clone of $repo_web_location/$reponame"
   # FIXME look in $repodir/.git/config to be sure 
   cd $repodirname
   #   undo the patches to RAJA
   if [ "$reponame" == "$AOMP_RAJA_REPO_NAME" ] ; then
      git checkout include/RAJA/policy/atomic_auto.hpp
      cd blt
      git checkout cmake/SetupCompilerOptions.cmake
      cd $repodirname
   fi
   if [ "$STASH_BEFORE_PULL" == "YES" ] ; then
      if [ "$reponame" != "$AOMP_RAJA_REPO_NAME" ] ; then
         git stash -u
      fi
   fi
   echo "git pull "
   git pull 
   echo "cd $repodirname ; git checkout $COBRANCH"
   git checkout $COBRANCH
   #echo "git pull "
   #git pull 
   if [ "$reponame" == "$AOMP_RAJA_REPO_NAME" ] ; then
     echo "git submodule update"
     git submodule update
     echo "git pull"
     git pull
   fi
else 
   echo --- NEW CLONE of repo $reponame to $repodirname ----
   cd $AOMP_REPOS_TEST
   if [[ "$reponame" == "$AOMP_RAJA_REPO_NAME" || "$reponame" == "$AOMP_RAJAPERF_REPO_NAME" ]]; then
     git clone --recursive -b $COBRANCH $repo_web_location/$reponame $reponame
   else
     echo git clone $repo_web_location/$reponame
     git clone $repo_web_location/$reponame $reponame
     echo "cd $repodirname ; git checkout $COBRANCH"
     cd $repodirname
     git checkout $COBRANCH
   fi
fi
cd $repodirname
echo git status
git status
}

mkdir -p $AOMP_REPOS_TEST

# ---------------------------------------
# The following repos is in AMDComputeLibraries
# ---------------------------------------
repo_web_location=$GITROCLIB
reponame=$AOMP_APPS_REPO_NAME
COBRANCH=$AOMP_APPS_REPO_BRANCH
clone_or_pull

repo_web_location=$GITSOLVV
reponame=$AOMP_SOLVV_REPO_NAME
COBRANCH=$AOMP_SOLVV_REPO_BRANCH
clone_or_pull

repo_web_location=$GITNEKBONE
reponame=$AOMP_NEKBONE_REPO_NAME
COBRANCH=$AOMP_NEKBONE_REPO_BRANCH
clone_or_pull

repo_web_location=$GITLLNLGOULASH
reponame=$AOMP_GOULASH_REPO_NAME
COBRANCH=$AOMP_GOULASH_REPO_BRANCH
clone_or_pull
