#!/bin/bash
#
#  list_test_repos.sh: Compare the repos in $AOMP_REPOS_TEST
#                      with the manifest for the test repos.
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=${AOMP_USE_CCACHE:-0}

. $thisdir/aomp_common_vars
# --- end standard header ----

EPSDB_LIST=${EPSDB_LIST:-"openmpapps sollve_vv Nekbone goulash fortran-babelstream babelstream OvO"}

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
   if [ "$actual_branch" == "(no" ] || [ "$actual_branch" == "(HEAD" ] ; then
      is_hash=1
      actual_hash=`git branch | awk '/\*/ { print $5; }' | cut -d")" -f1`
      if [ "$actual_hash" == "$branch_name" ] ; then
         WARNWORD="tagged"
	 thiscommit=$HASH
      elif [ "$actual_hash" != "$HASH" ] ; then
         WARNWORD="!BADHASH"
         thiscommit=$actual_hash
      else
         WARNWORD="hash"
         thiscommit=$actual_hash
      fi
   fi
   if [ "$branch_name" != "$actual_branch" ] && [ $is_hash == 0 ] ; then
      WARNWORD="!BRANCH!"
      printbranch=$actual_branch
   else
      printbranch=${REPO_RREV##*release/}
   fi
if [[ -f .git/config ]] ; then 

   url=`grep -m1 url .git/config | cut -d":" -f2- | cut -d"/" -f3-`
   project_name=`echo $url | cut -d"/" -f2- | tr '[:upper:]' '[:lower:]'`
   #website=`echo $url | cut -d"/" -f1`
   if [[ "$REPO_REMOTE" == "roc" ]] ; then
        manifest_project=`echo radeonopencompute/$REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   elif [[ "$REPO_REMOTE" == "roctools" ]] ; then
        manifest_project=`echo ROCM-Developer-Tools/$REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   elif [[ "$REPO_REMOTE" == "amdlibs" ]] ; then
        manifest_project=`echo AMDComputeLibraries/$REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   elif [[ "$REPO_REMOTE" == "omphost" ]] ; then
        manifest_project=`echo doru1004/$REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   elif [[ "$REPO_REMOTE" == "julia" ]] ; then
        manifest_project=`echo JuliaMath/$REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   elif [[ "$REPO_REMOTE" == "tapple" ]] ; then
        manifest_project=`echo TApplencourt/$REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   elif [[ "$REPO_REMOTE" == "gerritgit" ]] ; then
        manifest_project=`echo $REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   else
        manifest_project=`echo $REPO_REMOTE/$REPO_PROJECT | tr '[:upper:]' '[:lower:]'`
   fi
   #tr '[:upper:]' '[:lower:]'`
   if [[ "$manifest_project" != "$project_name" ]] ; then
      echo "WARNING Actual project  : $project_name"
      echo "        Manifest project: $manifest_project"
      WARNWORD="!REPO!"
   fi
   printf "%10s %12s %13s %17s %12s %10s %18s %18s %8s\n" $REPO_REMOTE $printbranch $REPO_PATH ${REPO_PROJECT} $thiscommit $thisdatevar "$author" "$forauthor" "$WARNWORD"
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

function clone_or_pull(){
repodirname=${AOMP_REPOS_TEST}/$reponame
echo
if [ -d $repodirname  ] ; then
   echo "--- Pulling updates to existing dir $repodirname ----"
   echo "    We assume this came from an earlier clone of $repo_web_location$repogitname"
   # FIXME look in $repodir/.git/config to be sure
   cd $repodirname
   #   undo the patches to RAJA
   if [ "$reponame" == "raja" ] ; then
      git checkout include/RAJA/policy/atomic_auto.hpp
      cd blt
      git checkout cmake/SetupCompilerOptions.cmake
      cd $repodirname
   fi
   if [ "$STASH_BEFORE_PULL" == "YES" ] ; then
      if [ "$reponame" != "raja" ] ; then
         git stash -u
      fi
   fi
   echo "git pull "
   git pull
   echo "cd $repodirname ; git checkout $COBRANCH"
   git checkout $COBRANCH
   #echo "git pull "
   #git pull
   if [ "$reponame" == "raja" ] ; then
     echo "git submodule update"
     git submodule update
     echo "git pull"
     git pull
   fi
else
   echo --- NEW CLONE of repo $reponame to $repodirname ----
   cd $AOMP_REPOS_TEST
   if [[ "$reponame" == "raja" || "$reponame" == "RAJAPerf" ]]; then
     git clone --recursive -b $COBRANCH $repo_web_location$repogitname $reponame
   else
     echo git clone $repo_web_location$repogitname
     git clone ${repo_web_location}${repogitname} $reponame
     echo "cd $repodirname ; git checkout $COBRANCH"
     cd $repodirname
     git checkout $COBRANCH
   fi
fi
cd $repodirname
echo git status
git status
}

manifest_file=$thisdir/../manifests/test_${AOMP_VERSION}.xml
if [ ! -f $manifest_file ] ; then 
   echo "ERROR manifest file missing: $manifest_file"
   exit 1
fi
# Manifest file must be one project line per repo
if [ "$1" == "list" ] ; then
   printf "MANIFEST FILE  : %66s\n" $manifest_file
   printf "AOMP_REPOS_TEST: %66s\n" $AOMP_REPOS_TEST
   printf "%10s %12s %13s %17s %12s %10s %18s %18s\n" "repo src" "branch" "path" "repo name" "last hash" "updated" "commitor" "for author"
   printf "%10s %12s %13s %17s %12s %10s %18s %18s\n" "--------" "------" "----" "---------" "---------" "-------" "--------" "----------"
fi

tmpfile=/tmp/mlines$$
cat $manifest_file | grep project > $tmpfile

if [ ! -d ~/git/aomp-test ]; then
  mkdir -p ~/git/aomp-test
fi

while read line ; do 
      line_is_good=1
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
         if [ -z "${field##*groups=*}" ]  ; then
	    groups=$(eval echo `echo $field | cut -d= -f2 `) 
         fi
      done
      reponame=$path
      repogitname=$name
      repodirname=$AOMP_REPOS_TEST/$reponame
      repo_web_location=`grep http $manifest_file | grep $remote | cut -d":" -f2 | cut -d"\"" -f1`
      repo_web_location="https:${repo_web_location}"
      REPO_PROJECT=$name
      REPO_PATH=$path
      REPO_RREV=$COBRANCH
      REPO_REMOTE=$remote
      if [ "$1" == "list" ] ; then
         if [ -d $repodirname ] ; then
            cd $repodirname
            list_repo_from_manifest
         else
            echo $repodirname not found
         fi
      else
         if [[ "$groups" != "skip" ]] ; then 
           if [ "$EPSDB" == 1 ]; then
             for suite in $EPSDB_LIST; do
               if [ "$reponame" == "$suite" ]; then
                 echo "clone_or_pull $repo_web_location PATH:$reponame $COBRANCH groups:$groups"
                 clone_or_pull
                 break
               fi
             done
           else
             echo "clone_or_pull $repo_web_location PATH:$reponame $COBRANCH groups:$groups"
             clone_or_pull
           fi
         fi
      fi
done <$tmpfile
rm $tmpfile

exit 
