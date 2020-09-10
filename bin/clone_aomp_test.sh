#!/bin/bash
#
#  clone_aomp.sh:  Clone the repositories needed to build the aomp compiler.  
#                  Currently AOMP needs 14 repositories.
#
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

# ---------------------------------------
# The following repo is for testing raja from LLNL
# ---------------------------------------
repo_web_location=$GITLLNL
reponame=$AOMP_RAJA_REPO_NAME
COBRANCH=$AOMP_RAJA_REPO_BRANCH
clone_or_pull

repo_web_location=$GITLLNL
reponame=$AOMP_RAJAPERF_REPO_NAME
COBRANCH=$AOMP_RAJAPERF_REPO_BRANCH
clone_or_pull

repo_web_location=$GITSOLVV
reponame=$AOMP_SOLVV_REPO_NAME
COBRANCH=$AOMP_SOLVV_REPO_BRANCH
clone_or_pull

repo_web_location=$GITNEKBONE
reponame=$AOMP_NEKBONE_REPO_NAME
COBRANCH=$AOMP_NEKBONE_REPO_BRANCH
clone_or_pull

repo_web_location=$GITOVO
reponame=$AOMP_OVO_REPO_NAME
COBRANCH=$AOMP_OVO_REPO_BRANCH
clone_or_pull

repo_web_location=$GITOMPTESTS
reponame=$AOMP_OMPTESTS_REPO_NAME
COBRANCH=$AOMP_OMPTESTS_REPO_BRANCH
clone_or_pull

repo_web_location=$GITQMCPACK
reponame=$AOMP_QMCPACK_REPO_NAME
COBRANCH=$AOMP_QMCPACK_REPO_BRANCH
clone_or_pull
