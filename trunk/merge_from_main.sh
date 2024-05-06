#!/bin/bash
#
# merge_from_main.sh:  Merge main branch into amd-trunk-dev2 (ATD2)
#
#       This script has no command line arguments. 
#       However, it requires that the environment variable variable
#       GIT_ADMIN is set to: <githubuserid>:<password>
#       unless GITPUSH is set to NO
#
# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/trunk_common_vars
# --- end standard header ----

GITPUSH=${GITPUSH:-NO}
if [ "$GITPUSH" == "YES" ] ; then
   if [ -z $GIT_ADMIN ] ; then
      echo "ERROR: You must set environment variable GIT_ADMIN to <userid>:<password>"
      echo "       or set GITPUSH=NO"
      exit 1
   fi
fi
REPO_MIRROR="https://github.com/ROCm/llvm-project"
REPO="REPO_MIRROR"
_dev_branch="amd-trunk-dev2"
logfile="/tmp/${USER}_trunk_merge.log"
merge_repo_dir=$TRUNK_REPOS/llvm-project
touch $logfile

if [ ! -f $logfile ] ; then 
   echo "ERROR: logfile $logfile missing"
   exit 1
fi

echo " " >> $logfile
echo "===============  START $0 ========================" | tee -a $logfile
date | tee -a $logfile
echo "==========================================" | tee -a $logfile

if [ ! -d $merge_repo_dir ] ; then 
   echo "ERROR: directory $merge_repo_dir missing" | tee -a $logfile
   exit 1
fi

# save the users current directory to return before exit
curdir=$PWD
echo "cd $merge_repo_dir" | tee -a $logfile
cd $merge_repo_dir

echo "git checkout main"  | tee -a $logfile
git checkout main  2>&1 | tee -a $logfile
if [ $? != 0 ] ; then 
   echo "ERROR: git checkout main failed" | tee -a  $logfile
   cd $curdir
   exit 1
fi

_gitdiff=`git diff`
if [ ! -z "$_gitdiff" ] ; then 
   echo "ERROR: The main branch must have no local changes" | tee -a $logfile
   echo "       git diff follows:" | tee -a $logfile
   echo "$_gitdiff"  | tee -a $logfile
   echo " "  | tee -a $logfile
   echo "Please remove all changes to main branch " | tee -a $logfile
   echo " "  | tee -a $logfile
   cd $curdir
   exit 1
fi

echo "git pull" | tee -a $logfile
git pull 2>&1 | tee -a $logfile
if [ $? != 0 ] ; then 
   echo "ERROR: git pull on main branch failed" | tee -a  $logfile
   cd $curdir
   exit 1
fi

echo "git checkout $_dev_branch " | tee -a $logfile
git checkout $_dev_branch 2>&1 | tee -a $logfile
if [ $? != 0 ] ; then 
   echo "ERROR: git checkout $_dev_branch failed" | tee -a  $logfile
   cd $curdir
   exit 1
fi
_gitdiff=`git diff`
if [ ! -z "$_gitdiff" ] ; then 
   echo "ERROR: Branch $_dev_branch must have no local changes" | tee -a $logfile
   echo "       git diff follows:" | tee -a $logfile
   echo "$_gitdiff"  | tee -a $logfile
   echo " "  | tee -a $logfile
   echo "Please save your changes and remove above differences in $_dev_branch" | tee -a $logfile
   echo " "  | tee -a $logfile
   cd $curdir
   exit 1
fi

echo "git pull" | tee -a $logfile
git pull 2>&1 | tee -a $logfile
if [ $? != 0 ] ; then 
   echo "ERROR: git pull on branch $_dev_branch failed" | tee -a  $logfile
   cd $curdir
   exit 1
fi

#  Check if $_dev_branch is behind main 
_commits_behind=`git rev-list --left-right --count main...$_dev_branch | cut -f1`
if [ $_commits_behind == 0 ] ; then 
   echo "WARNING: No need for $0 because $_dev_branch is not behind. Exiting" | tee -a $logile
   echo "===============  DONE $0 ====================" | tee -a  $logfile
   cd $curdir
   exit
fi

_fail=0
echo "git merge main " | tee -a $logfile
git merge main -m "Merge attempt by $0 into branch $_dev_branch " 2>&1 | tee -a $logfile
_merge_rc=$?
if [ $_merge_rc == 0 ] ; then 
  echo "git status | grep -A10 Unmerged"
  git status | grep -A10 "Unmerged"
  if [ $? == 0 ] ; then
    echo "ERROR:  Merge FAILED with unmerged files"
    _fail=1
  else
   if [ "$GITPUSH" == "YES" ] ; then
      echo "===============  Successful MERGE  ====================" | tee -a  $logfile
      echo "git push ${!REPO} $_dev_branch" | tee -a $logfile
      PUSH_LOG="$(git push ${!REPO/https:\/\//https:\/\/$GIT_ADMIN@} $_dev_branch 2>&1)"
      _push_rc=$?
      if [ $_push_rc == 0 ] ; then
         echo "===============  Successful push rc: $_push_rc =================" | tee -a  $logfile
      else
         echo " " | tee -a $logfile
         echo "ERROR: PUSH FAILED with rc $_push_rc  ====================" | tee -a  $logfile
         echo "       CHECK YOUR PASSWORD IN GIT_ADMIN" | tee -a $logfile
         echo "       GIT_ADMIN=$GIT_ADMIN"  # do not write a password to any log file
         _fail=1
      fi
   else
      echo "WARNING: SKIPPING git push because GITPUSH=$GITPUSH" | tee -a $logfile
   fi
  fi
else
   echo " " >> $logfile
   echo "ERROR: MERGE FAILED with rc $_merge_rc " | tee -a  $logfile
   echo "       See log, repair and push branch $_dev_branch" | tee -a  $logfile
   _fail=1
fi
echo " " >> $logfile
date | tee -a $logfile
if [ $_fail == 0 ] ; then 
  echo "===============  SUCCESS $0 ====================" | tee -a  $logfile
else
  echo "===============  FAIL $0 ====================" | tee -a  $logfile
fi
cd $curdir
