#!/bin/bash
#
#  diff_aomp.sh:  Diff the repositories needed to build the aomp compiler.  
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

function just_diff(){
repodirname=$AOMP_REPOS/$reponame
echo
if [ -d $repodirname  ] ; then 
   echo "--- Diffing $repodirname ----"
   # FIXME look in $repodir/.git/config to be sure 
   cd $repodirname
   git diff 
   if [ "$reponame" == "$AOMP_RAJA_REPO_NAME" ] ; then
     git diff
   fi
else 
   echo --- Diff of repo $reponame to $repodirname ----
   cd $AOMP_REPOS
   if [ "$reponame" == "$AOMP_RAJA_REPO_NAME" ] ; then
    echo Recursive diff needed - fixme
     git diff  -b $COBRANCH $repo_web_location/$reponame $reponame
   else
     echo git diff $repo_web_location/$reponame
     git diff $repo_web_location/$reponame $reponame
   fi
fi
echo git status
git status
}


# ---------------------------------------
#  The following REPOS are in ROCm-Development
# ---------------------------------------
repo_web_location=$GITROCDEV

reponame=$AOMP_REPO_NAME
COBRANCH=$AOMP_REPO_BRANCH
just_diff

reponame=$AOMP_EXTRAS_REPO_NAME
COBRANCH=$AOMP_EXTRAS_REPO_BRANCH
just_diff

reponame=$AOMP_PROJECT_REPO_NAME
COBRANCH=$AOMP_PROJECT_REPO_BRANCH
just_diff

reponame=$AOMP_FLANG_REPO_NAME
COBRANCH=$AOMP_FLANG_REPO_BRANCH
just_diff

reponame=$AOMP_HIPVDI_REPO_NAME
COBRANCH=$AOMP_HIPVDI_REPO_BRANCH
just_diff

# ---------------------------------------
# The following repos are in RadeonOpenCompute
# ---------------------------------------
repo_web_location=$GITROC

reponame=$AOMP_LIBDEVICE_REPO_NAME
COBRANCH=$AOMP_LIBDEVICE_REPO_BRANCH
just_diff

reponame=$AOMP_ROCT_REPO_NAME
COBRANCH=$AOMP_ROCT_REPO_BRANCH
just_diff

reponame=$AOMP_ROCR_REPO_NAME
COBRANCH=$AOMP_ROCR_REPO_BRANCH
just_diff

reponame=$AOMP_COMGR_REPO_NAME
COBRANCH=$AOMP_COMGR_REPO_BRANCH
just_diff

# ---------------------------------------
# The following repos is in AMDComputeLibraries
# ---------------------------------------
repo_web_location=$GITROCLIB
reponame=$AOMP_APPS_REPO_NAME
COBRANCH=$AOMP_APPS_REPO_BRANCH
just_diff

# ---------------------------------------
# The following repo is for testing raja from LLNL
# ---------------------------------------
repo_web_location=$GITLLNL
reponame=$AOMP_RAJA_REPO_NAME
COBRANCH=$AOMP_RAJA_REPO_BRANCH
just_diff

