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

if [ "$thisdir" != "$AOMP_REPOS/$AOMP_REPO_NAME/bin" ] ; then
   echo
   echo "ERROR:  This clone_aomp.sh script is found in directory $thisdir "
   echo "        But it should be found at $AOMP_REPOS/$AOMP_REPO_NAME/bin because the value"
   echo "        of AOMP_REPOS is $AOMP_REPOS. Either the environment variable AOMP_REPOS"
   echo "        is wrong or the $AOMP_REPO_NAME repository was cloned to the wrong directory. Consider"
   echo "        moving this $AOMP_REPO_NAME repository to $AOMP_REPOS/$AOMP_REPO_NAME (prefered)  OR"
   echo "        set the environment variable AOMP_REPOS to the parent directory of the $AOMP_REPO_NAME"
   echo "        repository before running $0"
   echo
   exit 1
fi

function list_repo(){
repodirname=$AOMP_REPOS/$reponame
cd $repodirname
abranch=`git branch | awk '/\*/ { print $2; }'`
echo `git config --get remote.origin.url` " desired: " $COBRANCH " actual: " $abranch "  " `git log --numstat --format="%h" |head -1`
}

function clone_or_pull(){
if [ "$LISTONLY" == 'list' ]; then
list_repo
return
fi

repodirname=$AOMP_REPOS/$reponame
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
   cd $AOMP_REPOS
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

if [[ "$AOMP_VERSION" == "13.1" ]] || [[ $AOMP_MAJOR_VERSION -gt 13 ]] ; then
   # For 13.1 and beyond, initialization requires init_aomp_repos.sh to be run
   # once. Therefore a full repo sync is not necessary and it could get in the
   # way of unstaged or uncommited work. So all we want to do here is to update
   # each repo with a "git pull" in each repo directory even though the name
   # of this script is clone_aomp.sh for historical reasons.
   repobindir=$AOMP_REPOS/.bin
   if [ "$1" == "list" ] ; then
      tmpfile=/tmp/repostats$$
      $repobindir/repo forall -g unlocked -c $thisdir/repo_forall_cmds list | sort >$tmpfile
      $repobindir/repo forall -g revlocked -c $thisdir/repo_forall_cmds list >>$tmpfile
      cat $tmpfile
      rm $tmpfile
   else
      echo $repobindir/repo forall -p -g unlocked -c \'git pull\'
      $repobindir/repo forall -p -g unlocked -c 'git pull'
   fi
   rc=$?
   exit $rc
fi

mkdir -p $AOMP_REPOS

# ---------------------------------------
#  The following REPOS are in ROCm-Development
# ---------------------------------------

# LLVM handled for 11 vs 12
LISTONLY=$1

repo_web_location=$GITPROJECT
reponame=$AOMP_PROJECT_REPO_NAME
repogitname=$AOMP_PROJECT_REPO_NAME
COBRANCH=$AOMP_PROJECT_REPO_BRANCH
clone_or_pull


repo_web_location=$GITROCDEV

reponame=$AOMP_REPO_NAME
repogitname=$AOMP_REPO_NAME
COBRANCH=$AOMP_REPO_BRANCH
if [ "$LISTONLY" == 'list' ]; then
list_repo
#clone_or_pull
fi


reponame=$AOMP_EXTRAS_REPO_NAME
repogitname=$AOMP_EXTRAS_REPO_NAME
COBRANCH=$AOMP_EXTRAS_REPO_BRANCH
COSHAKEY=$AOMP_EXTRAS_REPO_SHA
clone_or_pull
COSHAKEY=""

reponame=$AOMP_FLANG_REPO_NAME
repogitname=$AOMP_FLANG_REPO_NAME
COBRANCH=$AOMP_FLANG_REPO_BRANCH
clone_or_pull

# ---------------------------------------
# The following repos are in RadeonOpenCompute
# ---------------------------------------
repo_web_location=$GITROC
reponame=$AOMP_LIBDEVICE_REPO_NAME
repogitname=$AOMP_LIBDEVICE_REPO_NAME
COBRANCH=$AOMP_LIBDEVICE_REPO_BRANCH
COSHAKEY=$AOMP_LIBDEVICE_REPO_SHA
clone_or_pull
COSHAKEY=""

reponame=$AOMP_ROCT_REPO_NAME
repogitname=$AOMP_ROCT_REPO_NAME
COBRANCH=$AOMP_ROCT_REPO_BRANCH
clone_or_pull

reponame=$AOMP_ROCR_REPO_NAME
repogitname=$AOMP_ROCR_REPO_NAME
COBRANCH=$AOMP_ROCR_REPO_BRANCH
clone_or_pull

reponame=$AOMP_COMGR_REPO_NAME
repogitname=$AOMP_COMGR_REPO_NAME
COBRANCH=$AOMP_COMGR_REPO_BRANCH
COSHAKEY=$AOMP_COMGR_REPO_SHA
clone_or_pull
COSHAKEY=""

reponame=$AOMP_RINFO_REPO_NAME
repogitname=$AOMP_RINFO_REPO_NAME
COBRANCH=$AOMP_RINFO_REPO_BRANCH
COSHAKEY=$AOMP_RINFO_REPO_SHA
clone_or_pull
COSHAKEY=""

repo_web_location=$GITROCDEV
reponame=$AOMP_VDI_REPO_NAME
repogitname=$AOMP_VDI_REPO_GITNAME
COBRANCH=$AOMP_VDI_REPO_BRANCH
clone_or_pull
repo_web_location=$GITROCDEV
reponame=$AOMP_HIPVDI_REPO_NAME
repogitname=$AOMP_HIPVDI_REPO_GITNAME
COBRANCH=$AOMP_HIPVDI_REPO_BRANCH
clone_or_pull
repo_web_location=$GITROC
reponame=$AOMP_OCL_REPO_NAME
repogitname=$AOMP_OCL_REPO_GITNAME
COBRANCH=$AOMP_OCL_REPO_BRANCH
clone_or_pull
if [ "$AOMP_BUILD_DEBUG" == "1" ] ; then
#   repo_web_location="git://sourceware.org/git"
   repo_web_location=$GITROCDEV
   reponame=$AOMP_GDB_REPO_NAME
   repogitname=$AOMP_GDB_REPO_NAME
   COBRANCH=$AOMP_GDB_REPO_BRANCH
   clone_or_pull
   reponame=$AOMP_DBGAPI_REPO_NAME
   repogitname=$AOMP_DBGAPI_REPO_NAME
   COBRANCH=$AOMP_DBGAPI_REPO_BRANCH
   clone_or_pull
   reponame=$AOMP_TRACE_REPO_NAME
   repogitname=$AOMP_TRACE_REPO_NAME
   COBRANCH=$AOMP_TRACE_REPO_BRANCH
   clone_or_pull
   reponame=$AOMP_PROF_REPO_NAME
   repogitname=$AOMP_PROF_REPO_NAME
   COBRANCH=$AOMP_PROF_REPO_BRANCH
   clone_or_pull
fi
