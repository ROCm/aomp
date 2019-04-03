#!/bin/bash
#
#  clone_trunk.sh:  Clone the repositories needed to build the OPENMP clang/llvm trunk
#                   like build of AOMP but only 4 components and use the master branch
#                 
#
export AOMP_REPOS=$HOME/git/trunk
export AOMP_LLVM_REPO_BRANCH=master
export AOMP_LLD_REPO_BRANCH=master
export AOMP_CLANG_REPO_BRANCH=master
export AOMP_OPENMP_REPO_BRANCH=master

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
      git stash -u
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

reponame=$AOMP_OPENMP_REPO_NAME
COBRANCH=$AOMP_OPENMP_REPO_BRANCH
clone_or_pull

reponame=$AOMP_LLVM_REPO_NAME
COBRANCH=$AOMP_LLVM_REPO_BRANCH
clone_or_pull

reponame=$AOMP_CLANG_REPO_NAME
COBRANCH=$AOMP_CLANG_REPO_BRANCH
clone_or_pull

reponame=$AOMP_LLD_REPO_NAME
COBRANCH=$AOMP_LLD_REPO_BRANCH
clone_or_pull

