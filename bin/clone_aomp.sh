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
   if [ $? == 0 ] ; then 
      echo "cd $repodirname ; git checkout $COBRANCH"
      cd $repodirname
      git checkout $COBRANCH
   fi
fi 
if [ -d $repodirname ] ; then 
   echo cd $repodirname
   cd $repodirname
   if [ "$COSHAKEY" != "" ] ; then
     echo git checkout $COSHAKEY
     git checkout $COSHAKEY
   fi
   echo git status
   git status
fi
}

function list_repo_from_manifest(){
   logcommit=`git log -1 | grep "^commit" | cut -d" " -f2 | xargs`
   thiscommit=${logcommit:0:12}
   thisdate=`git log -1 --pretty=fuller | grep "^CommitDate:" | cut -d":" -f2- | xargs | cut -d" " -f2-`
   get_monthnumber $thisdate
   thisday=`echo $thisdate | cut -d" " -f2`
   thisyear=`echo $thisdate | cut -d" " -f4`
   printf -v thisdatevar "%4u-%2s-%02u" $thisyear $monthnumber $thisday
   author=`git log -1 --pretty=fuller | grep "^Commit:" | cut -d":" -f2- | cut -d"<" -f1 | xargs`
   repodirname=$REPO_PATH
   HASH=`git log -1 --numstat --format="%h" | head -1`
   is_hash=0
   branch_name=${REPO_RREV}
   # get the actual branch
   actual_branch=`git branch | awk '/\*/ { print $2; }'`
   rc=0
   if [ "$actual_branch" == "(no" ] || [ "$actual_branch" == "(HEAD" ] ; then
      is_hash=1
      actual_hash=`git branch | awk '/\*/ { print $5; }' | cut -d")" -f1`
      if [ "$actual_hash" != "$HASH" ] ; then
          rc=1
      fi
   fi
   if [ "$branch_name" != "$actual_branch" ] && [ $is_hash == 0 ] ; then
      rc=2
   fi
   if [ $rc == 1 ] ; then
      printf "%24s %20s %12s %10s %26s %20s %8s\n" $actual_hash $REPO_PATH $thiscommit $thisdatevar ${REPO_PROJECT} "$author" "!BADHASH!"
   elif [ $is_hash  == 1 ] ; then
      printf "%24s %20s %12s %10s %26s %20s\n" $actual_hash $REPO_PATH $thiscommit $thisdatevar ${REPO_PROJECT} "$author"
   elif [ $rc == 2 ] ; then
      printf "%24s %20s %12s %10s %26s %20s %8s\n" $actual_branch $REPO_PATH $thiscommit $thisdatevar ${REPO_PROJECT} "$author" "!BRANCH!"
   else
      printbranch=${REPO_RREV##*release/}
      printf "%10s %12s %20s %12s %10s %31s %18s \n" $REPO_REMOTE $printbranch $REPO_PATH $thiscommit $thisdatevar ${REPO_PROJECT} "$author"
   fi
}

function get_monthnumber() {
    case $(echo ${1:0:3} | tr '[a-z]' '[A-Z]') in
        JAN) monthnumber="01" ;;
        FEB) monthnumber="02" ;;
        MAR) monthnumber="03" ;;
        APR) monthnumber="04" ;;
        MAY) monthnumber="05" ;;
        JUN) monthnumber="06" ;;
        JUL) monthnumber="07" ;;
        AUG) monthnumber="08" ;;
        SEP) monthnumber="09" ;;
        OCT) monthnumber="10" ;;
        NOV) monthnumber="11" ;;
        DEC) monthnumber="12" ;;
    esac
}

if [[ "$AOMP_VERSION" == "13.1" ]] || [[ $AOMP_MAJOR_VERSION -gt 13 ]] ; then
   # For 13.1 and beyond, we use a manifest file to specify the repos to clone.
   # However, we gave up on using the repo command to clone the repos. 
   # That is all done here by parsing the manifest file. 
   manifest_file=$thisdir/../manifests/aomp_${AOMP_VERSION}.xml
   if [ ! -f $manifest_file ] ; then 
      echo "ERROR manifest file missing: $manifest_file"
      exit 1
   fi
   tmpfile=/tmp/mlines$$
   # Manifest file must be one project line per repo
   cat $manifest_file | grep project > $tmpfile
   if [ "$1" == "list" ] ; then
      printf "%10s %12s %20s %12s %10s %31s %18s \n" "repo src" "branch" "path" "last hash" "updated" "repo name" "last author"
      printf "%10s %12s %20s %12s %10s %31s %18s \n" "--------" "------" "----" "---------" "-------" "---------" "-----------"
   fi
   while read line ; do 
      remote=`echo $line | grep remote | cut -d"=" -f2`
      for field in `echo $line` ; do 
         if [ -z "${field##*remote=*}" ]  ; then
	    # strip off = and double quotes 
	    remote=$(eval echo `echo $field | cut -d= -f2 `) 
         fi
         if [ -z "${field##*name=*}" ]  ; then
	    name=$(eval echo `echo $field | cut -d= -f2 `) 
         fi
         if [ -z "${field##*path=*}" ]  ; then
	    path=$(eval echo `echo $field | cut -d= -f2 `) 
         fi
         if [ -z "${field##*revision=*}" ]  ; then
	    COBRANCH=$(eval echo `echo $field | cut -d= -f2 `) 
         fi
      done
      reponame=$path
      repogitname=$name
      if [ $remote == "roc" ] ; then 
         repo_web_location=$GITROC
      elif [ $remote == "roctools" ] ; then 
         repo_web_location=$GITROCDEV
      elif [ $remote == "gerritgit" ] ; then 
         repo_web_location=$GITGERRIT
      else
         echo sorry remote=$remote
      fi
      if [ "$1" == "list" ] ; then
         repodirname=$AOMP_REPOS/$reponame
	 if [ -d $repodirname ] ; then 
            REPO_PROJECT=$name
            REPO_PATH=$path
            REPO_RREV=$COBRANCH
	    REPO_REMOTE=$remote
            cd $repodirname
            list_repo_from_manifest
         fi
      else
	 if [ $reponame == "aomp" ] ; then 
            echo
            echo "Skipping pull of aomp repo "
	    echo
	 else
            clone_or_pull
         fi
      fi
   done <$tmpfile
   rm $tmpfile

   # build_rocr.sh expects directory rocr-runtime which is a subdir of hsa-runtime
   # Link in the open source hsa-runtime as "src" directory
   if [ -d $AOMP_REPOS/hsa-runtime ] ; then
      if [ ! -L $AOMP_REPOS/rocr-runtime/src ] ; then
         echo "Fixing rocr-runtime with correct link to hsa-runtime/opensrc/hsa-runtime src"
         echo mkdir -p $AOMP_REPOS/rocr-runtime
         mkdir -p $AOMP_REPOS/rocr-runtime
         echo cd $AOMP_REPOS/rocr-runtime
         cd $AOMP_REPOS/rocr-runtime
         echo ln -sf -t $AOMP_REPOS/rocr-runtime ../hsa-runtime/opensrc/hsa-runtime
         ln -sf -t $AOMP_REPOS/rocr-runtime ../hsa-runtime/opensrc/hsa-runtime
         echo ln -sf hsa-runtime src
         ln -sf hsa-runtime src
      fi
   fi
   exit $rc
fi

## Before 13.1 repos were specified with environment variablse in aomp_common_vars
#  
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
