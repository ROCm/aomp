#!/bin/bash
#
#  clone_f18.sh:  Clone the repositories needed to build the f18 compiler for amd f18 developers
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
. $thisdir/f18_common_vars
# --- end standard header ----

if [ "$thisdir" != "$F18_REPOS/$AOMP_REPO_NAME/f18bin" ] ; then
   echo
   echo "ERROR:  This clone_f18.sh script is found in directory $thisdir "
   echo "        But it should be found at $F18_REPOS/$AOMP_REPO_NAME/f18bin because the value"
   echo "        of F18_REPOS is $F18_REPOS. Either the environment variable F18_REPOS"
   echo "        is wrong or the $AOMP_REPO_NAME repository was cloned to the wrong directory. Consider"
   echo "        moving this $AOMP_REPO_NAME repository to $F18_REPOS/$AOMP_REPO_NAME (prefered)  OR"
   echo "        set the environment variable F18_REPOS to the parent directory of the $AOMP_REPO_NAME"
   echo "        repository before running $0"
   echo
   exit 1
fi

function list_repo(){
repodirname=$F18_REPOS/$reponame
cd $repodirname
echo `git config --get remote.origin.url` "  " $COBRANCH "  " `git log --numstat --format="%h" |head -1`
}

function clone_or_pull(){
if [ "$LISTONLY" == 'list' ]; then
list_repo
return
fi

repodirname=$F18_REPOS/$reponame
echo
if [ -d $repodirname  ] ; then 
   echo "--- Pulling updates to existing dir $repodirname ----"
   echo "    We assume this came from an earlier clone of $repo_web_location/$repogitname"
   # FIXME look in $repodir/.git/config to be sure 
   cd $repodirname
   if [ "$STASH_BEFORE_PULL" == "YES" ] ; then
     git stash -u
   fi
   echo "git pull "
   git pull
   echo "cd $repodirname ; git checkout $COBRANCH"
   git checkout $COBRANCH
   echo "git pull "
   git pull
else 
   echo --- NEW CLONE of repo $repogitname to $repodirname ----
   cd $F18_REPOS
   echo git clone -b $COBRANCH $repo_web_location/$repogitname $reponame
   git clone -b $COBRANCH $repo_web_location/$repogitname $reponame
   echo "cd $repodirname ; git checkout $COBRANCH"
   cd $repodirname
   git checkout $COBRANCH
fi
if [ "$COSHAKEY" != "" ] ; then
  git checkout $COSHAKEY
fi

cd $repodirname
echo git status
git status
}

mkdir -p $F18_REPOS

# ---------------------------------------
#  The following REPOS are in ROCm-Development
# ---------------------------------------
repo_web_location=$GITROCDEV

reponame=$AOMP_REPO_NAME
repogitname=$AOMP_REPO_NAME
COBRANCH=$AOMP_REPO_BRANCH
LISTONLY=$1
if [ "$LISTONLY" == 'list' ]; then
list_repo
# You must manually pull or create the aomp repository 
#clone_or_pull
fi

reponame=$F18_PROJECT_REPO_NAME
repogitname=$F18_PROJECT_REPO_NAME
COBRANCH=$F18_PROJECT_REPO_BRANCH
clone_or_pull

reponame=$F18_EXTRAS_REPO_NAME
repogitname=$F18_EXTRAS_REPO_NAME
COBRANCH=$F18_EXTRAS_REPO_BRANCH
clone_or_pull

reponame=$F18_FLANG_REPO_NAME
repogitname=$F18_FLANG_REPO_NAME
COBRANCH=$F18_FLANG_REPO_BRANCH
clone_or_pull

# ---------------------------------------
# The following repos are in RadeonOpenCompute
# ---------------------------------------
repo_web_location=$GITROC

reponame=$F18_ROCT_REPO_NAME
repogitname=$F18_ROCT_REPO_NAME
COBRANCH=$F18_ROCT_REPO_BRANCH
clone_or_pull

reponame=$F18_ROCR_REPO_NAME
repogitname=$F18_ROCR_REPO_NAME
COBRANCH=$F18_ROCR_REPO_BRANCH
clone_or_pull

