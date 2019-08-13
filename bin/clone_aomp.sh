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
repodirname=$AOMP_REPOS/$reponame
echo
if [ -d $repodirname  ] ; then 
   echo "--- Pulling updates to existing dir $repodirname ----"
   echo "    We assume this came from an earlier clone of $repo_web_location/$reponame"
   # FIXME look in $repodir/.git/config to be sure 
   cd $repodirname
   if [ "$STASH_BEFORE_PULL" == "YES" ] ; then
      if [ "$reponame" != "$AOMP_HCC_REPO_NAME" ] ; then
         git stash -u
      fi
   fi
   echo "git pull "
   git pull 
   echo "cd $repodirname ; git checkout $COBRANCH"
   git checkout $COBRANCH
   #echo "git pull "
   #git pull 
   if [ "$reponame" == "$AOMP_HCC_REPO_NAME" ] ; then
     echo "git submodule update"
     git submodule update
     echo "git pull"
     git pull
   fi
else 
   echo --- NEW CLONE of repo $reponame to $repodirname ----
   cd $AOMP_REPOS
   if [ "$reponame" == "$AOMP_HCC_REPO_NAME" ] ; then
     git clone --recursive -b $COBRANCH $repo_web_location/$reponame $reponame
   else
     echo git clone $repo_web_location/$reponame
     git clone $repo_web_location/$reponame $reponame
     echo "cd $repodirname ; git checkout $COBRANCH"
     cd $repodirname
     git checkout $COBRANCH
   fi
fi
echo git status
git status
}

mkdir -p $AOMP_REPOS

# ---------------------------------------
#  The following REPOS are in ROCm-Development
# ---------------------------------------
repo_web_location=$GITROCDEV

reponame=$AOMP_REPO_NAME
COBRANCH=$AOMP_REPO_BRANCH
#clone_or_pull

reponame=$AOMP_EXTRAS_REPO_NAME
COBRANCH=$AOMP_EXTRAS_REPO_BRANCH
clone_or_pull

reponame=$AOMP_PROJECT_REPO_NAME
COBRANCH=$AOMP_PROJECT_REPO_BRANCH
clone_or_pull

reponame=$AOMP_FLANG_REPO_NAME
COBRANCH=$AOMP_FLANG_REPO_BRANCH
clone_or_pull

reponame=$AOMP_HIP_REPO_NAME
COBRANCH=$AOMP_HIP_REPO_BRANCH
clone_or_pull

# ---------------------------------------
# The following repos are in RadeonOpenCompute
# ---------------------------------------
repo_web_location=$GITROC

reponame=$AOMP_LIBDEVICE_REPO_NAME
COBRANCH=$AOMP_LIBDEVICE_REPO_BRANCH
clone_or_pull

reponame=$AOMP_ROCT_REPO_NAME
COBRANCH=$AOMP_ROCT_REPO_BRANCH
clone_or_pull

reponame=$AOMP_ROCR_REPO_NAME
COBRANCH=$AOMP_ROCR_REPO_BRANCH
clone_or_pull

reponame=$AOMP_ATMI_REPO_NAME
COBRANCH=$AOMP_ATMI_REPO_BRANCH
clone_or_pull

reponame=$AOMP_HCC_REPO_NAME
COBRANCH=$AOMP_HCC_REPO_BRANCH
clone_or_pull

reponame=$AOMP_COMGR_REPO_NAME
COBRANCH=$AOMP_COMGR_REPO_BRANCH
clone_or_pull

# ---------------------------------------
# The following repos is in AMDComputeLibraries
# ---------------------------------------
repo_web_location=$GITROCLIB
reponame=$AOMP_APPS_REPO_NAME
COBRANCH=$AOMP_APPS_REPO_BRANCH
clone_or_pull

# ---------------------------------------
# The following repo is internal to AMD
# ---------------------------------------
ping -c 1 $AOMP_INTERNAL_IP
if [ $? == 0 ] ; then
   echo
   echo " +---------------------------------------------------------------"
   echo " |  WARNING: USE YOUR AMD USERID AND PASSWORD TO CLONE FROM $GITINTERNAL"
   echo " +---------------------------------------------------------------"
   echo
   repo_web_location=$GITINTERNAL
   reponame=$AOMP_IAPPS_REPO_NAME
   COBRANCH=$AOMP_IAPPS_REPO_BRANCH
   clone_or_pull
fi
