#!/bin/bash
#
#  clone_aomp.sh:  Clone the repositories needed to build the aomp compiler.  
#                  Currently AOMP needs 14 repositories.
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=${AOMP_USE_CCACHE:-0}
. $thisdir/aomp_common_vars
# --- end standard header ----

thischk="$AOMP_REPOS/$AOMP_REPO_NAME/bin"
thischk=`realpath $thischk`
if [ "$thisdir" != "$thischk" ] ; then
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
   if [ $? != 0 ] && [ "$IGNORE_GIT_ERROR" != 1 ] ; then
     echo "git pull failed for: $repodirname"
     exit 1
   fi
   echo "cd $repodirname ; git checkout $COBRANCH"
   git checkout $COBRANCH
   if [ $? != 0 ] && [ "$IGNORE_GIT_ERROR" != 1 ] ; then
     echo "git checkout failed for: $repodirname"
     exit 1
   fi
   echo "git pull "
   git pull
   if [ $? != 0 ] && [ "$IGNORE_GIT_ERROR" != 1 ]; then
     echo "git pull failed for: $repodirname"
     exit 1
   fi
else 
   echo --- NEW CLONE of repo $repogitname to $repodirname ----
   cd $AOMP_REPOS
   if [ "$SINGLE_BRANCH" == 1 ]; then
     echo git clone -b $COBRANCH --depth=1 --single-branch $repo_web_location/$repogitname $reponame
     git clone -b $COBRANCH --depth=1 --single-branch $repo_web_location/$repogitname $reponame
   else
     echo git clone -b $COBRANCH $repo_web_location/$repogitname $reponame
     git clone -b $COBRANCH $repo_web_location/$repogitname $reponame
   fi
   if [ $? != 0 ] ; then
     echo "git clone failed for: $repodirname"
     exit 1
   else
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
   forauthor=`git log -1 --pretty=fuller | grep "^Author:" | cut -d":" -f2- | cut -d"<" -f1 | xargs`
   repodirname=$REPO_PATH
   HASH=`git log -1 --numstat --format="%h" | head -1`
   is_hash=0
   branch_name=${REPO_RREV}
   # get the actual branch
   actual_branch=`git branch | awk '/\*/ { print $2; }'`
   WARNWORD=""
   if [ "$actual_branch" == "(no" ] || [ "$actual_branch" == "(HEAD" ] || [ "$actual_branch" == "(detached" ] ; then
      is_hash=1
      WARNWORD="hash"
      git describe --exact-match --tags > /dev/null 2>&1
      # Repo has a tag checked out
      if [ $? -eq 0 ]; then
        head_tags=`git tag --points-at HEAD`
        # If tag is found in list of tags at HEAD, then it is correct.
        if [[ "$head_tags" =~ "$branch_name" ]]; then
          WARNWORD="tag"
          thiscommit=$branch_name
        else
          WARNWORD="!BADTAG"
          thiscommit=$HASH
        fi
      else # This is a hash not a tag
        actual_hash=`git branch | awk '/\*/ { print $5; }' | cut -d")" -f1`
        # RHEL 7 'git branch' returns (detached from 123456), try to get hash again.
        if [ "$actual_hash" == "" ] ; then
          actual_hash=`git branch | awk '/\*/ { print $4; }' | cut -d")" -f1`
        fi
        revision_regex="^$actual_hash"
        if [[ ! "$COSHAKEY" =~ $revision_regex ]] ; then
          WARNWORD="!BADHASH"
        fi
        thiscommit=$actual_hash
      fi
   fi
   if [ "$branch_name" != "$actual_branch" ] && [ $is_hash == 0 ] ; then
      WARNWORD="!BRANCH!"
      printbranch=$actual_branch
   else
      printbranch=${REPO_RREV##*release/}
   fi
   url=`grep url .git/config | cut -d":" -f2- | cut -d"/" -f3-`
   project_name=`echo $url | cut -d"/" -f2- | tr '[:upper:]' '[:lower:]'`
   #website=`echo $url | cut -d"/" -f1`
   if [[ "$REPO_REMOTE" == "roc" ]] ; then
        manifest_project=`echo ROCm/$REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   elif [[ "$REPO_REMOTE" == "roctools" ]] ; then
        manifest_project=`echo ROCm/$REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   elif [[ "$REPO_REMOTE" == "rocsw" ]] ; then
        manifest_project=`echo ROCm/$REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   elif [[ "$REPO_REMOTE" == "gerritgit" ]] ; then
        manifest_project=`echo $REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   elif [[ "$REPO_REMOTE" == "hwloc" ]] ; then
        manifest_project=`echo open-mpi/$REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   else
        manifest_project=`echo $REPO_REMOTE/$REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   fi
   #tr '[:upper:]' '[:lower:]'`
   if [[ "$manifest_project" != "$project_name" ]] ; then
      echo "WARNING Actual project  : $project_name"
      echo "        Manifest project: $manifest_project"
      WARNWORD="!REPO!"
   fi
   printf "%10s %12s %20s %25s %12s %10s %18s %18s %8s\n" $REPO_REMOTE $printbranch $REPO_PATH ${REPO_PROJECT} $thiscommit $thisdatevar "$author" "$forauthor" "$WARNWORD"
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
   ping -c 1 $AOMP_GIT_INTERNAL_IP 2> /dev/null >/dev/null
   if [ $? == 0 ] && [ "$AOMP_EXTERNAL_MANIFEST" != 1 ]; then
      # AMD internal repo file
      if [ "$AOMP_NEW" == "1" ]; then
         manifest_file=$thisdir/../manifests/aompi_new_${AOMP_VERSION}.xml
      else
         manifest_file=$thisdir/../manifests/aompi_${AOMP_VERSION}.xml
      fi
   else
      abranch=`git branch | awk '/\*/ { print $2; }'`
      # Use release manifest if on release branch
      if [ "$abranch" == "aomp-${AOMP_VERSION_STRING}" ]; then
         manifest_file=$thisdir/../manifests/aomp_${AOMP_VERSION_STRING}.xml
      else
	 if [ "$AOMP_NEW" == "1" ]; then
            manifest_file=$thisdir/../manifests/aomp_new_${AOMP_VERSION}.xml
         else
            manifest_file=$thisdir/../manifests/aomp_${AOMP_VERSION}.xml
	 fi
      fi
   fi
   echo "USED manifest file: $manifest_file"
   if [ ! -f $manifest_file ] ; then 
      echo "ERROR manifest file missing: $manifest_file"
      exit 1
   fi
   tmpfile=/tmp/mlines$$
   # Manifest file must be one project line per repo
   cat $manifest_file | grep project > $tmpfile
   if [ "$1" == "list" ] ; then
      printf "MANIFEST FILE: %40s\n" $manifest_file
      printf "%10s %12s %20s %25s %12s %10s %18s %18s\n" "repo src" "branch" "path" "repo name" "last hash" "updated" "commitor" "for author"
      printf "%10s %12s %20s %25s %12s %10s %18s %18s\n" "--------" "------" "----" "---------" "---------" "-------" "--------" "----------"
   fi
   while read line ; do 
      line_is_good=1
      remote=`echo $line | grep remote | cut -d"=" -f2`
      sha_key_used=0
      COSHAKEY=""
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
         if [ -z "${field##*upstream=*}" ]  ; then
           COBRANCH=$(eval echo `echo $field | cut -d= -f2 `)
           sha_key_used=1
         fi
         if [ -z "${field##*revision=*}" ] && [ "$sha_key_used" == 1 ]  ; then
           COSHAKEY=$(eval echo `echo $field | cut -d= -f2 `)
         elif [ -z "${field##*revision=*}" ]; then
           COBRANCH=$(eval echo `echo $field | cut -d= -f2 `)
         fi
      done
      reponame=$path
      repogitname=$name
      if [ "$remote" == "roc" ] ; then
         repo_web_location=$GITROC
      elif [ "$remote" == "roctools" ] ; then
         repo_web_location=$GITROCDEV
      elif [ "$remote" == "rocsw" ] ; then
         repo_web_location=$GITROCSW
      elif [ "$remote" == "gerritgit" ] ; then
         repo_web_location=$GITGERRIT
      elif [ "$remote" == "hwloc" ] ; then
         repo_web_location=$GITHWLOC
      else
         line_is_good=0
      fi
      if [ $line_is_good  == 1 ] ; then
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
      fi  # end line_is_good
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
echo "ERROR:  $0 no longer supports AOMP_VERSION $AOMP_VERSION "
exit 1 

