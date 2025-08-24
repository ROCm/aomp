#!/bin/bash
#
#  tr_set_aomp_branches_orig.sh: Change branches of submodules on original 
#                                clone of TheRock. 
#     
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/tr_aomp_common_vars"
# --- end standard header ----
 
_therockorig=$TR_AOMP_REPOS/TheRock.orig

_curdir=$PWD

cd $_therockorig
_aomp_repos_temp=()
for _line in `$TR_AOMP_REPOS/aomp/tr_aomp/tr_list_aomp_branches.sh | awk '{print $2 ":" $4}' | tr -d '"' ` ; do 
   _aomp_repos_temp+=($_line)
done

# Fix certain aomp reponames, use theRock reponames 
aomp_repos=()
for _repo in ${_aomp_repos_temp[@]} ; do
   _aomp_branch_name=${_repo%:*}
   _aomp_reponame=${_repo#*:}
   if [ "$_aomp_reponame" == "hip" ] ; then
      _aomp_reponame="HIP"
   elif [ "$_aomp_reponame" == "llvm-project" ] ; then
      _aomp_reponame="amd-llvm"
   elif [ "$_aomp_reponame" == "rocprof-trace-decoder" ] ; then
      _aomp_reponame="rocprof-trace-decoder/binaries"
   fi
   aomp_repos+=("$_aomp_branch_name:$_aomp_reponame")
done

submodules=()
for _line in `git submodule | awk '{print $2}'` ; do 
   _category=`echo $_line | cut -d"/" -f1`
   if [ -d $_category ] ; then 
      _dir=$_line
      _reponame=${_dir##*/}
      if [ "$_reponame" == "binaries" ] ; then 
        _reponame=${_dir#*/}
      fi
      # search for the branch we want for this submodule
      _aomp_branch_name=""
      for _repo in ${aomp_repos[@]} ; do
         _aomp_reponame=${_repo#*:}
	 #echo comparing  $_reponame to $_aomp_reponame for _repo $_repo FOR $_line
	 if [ $_reponame == $_aomp_reponame ] ; then 
           _aomp_branch_name=${_repo%:*}
	 fi
      done
      if [ -z "$_aomp_branch_name" ] ; then 
         echo "WARNING: NO AOMP REPO FOR SUBMODULE:$_reponame $_dir"
      else
         submodules+=($_reponame:$_category:$_dir:$_aomp_branch_name)
      fi
   fi
done 
echo 
for _repo in ${aomp_repos[@]} ; do
   _aomp_reponame=${_repo#*:}
   _aomp_submodule_entry=""
   for _submodule in ${submodules[@]} ; do
      _reponame=`echo $_submodule | cut -d":" -f1`
      if [ "$_aomp_reponame" == "$_reponame" ] ; then 
         _aomp_submodule_entry=_submodule
      fi
   done
   if [ -z "$_aomp_submodule_entry" ] ; then 
      echo "WARNING: NO SUBMODULE FOR AOMP REPO $_aomp_reponame"
   fi
done
echo 

for _submodule in ${submodules[@]} ; do
   _reponame=`echo $_submodule | cut -d":" -f1`
   _category=`echo $_submodule | cut -d":" -f2`
   _dir=`echo $_submodule | cut -d":" -f3`
   _aomp_branch_name=`echo $_submodule | cut -d":" -f4`
   echo
   echo "====== $_reponame $_dir $_aomp_branch_name"
   echo cd $_therockorig/$_dir
   cd $_therockorig/$_dir
   _full_branch_name=`git branch -a | grep -v "HEAD" | grep -v "release-staging" | grep -v "\*" | grep -m1 $_aomp_branch_name | xargs`
   echo " === _full_branch_name=$_full_branch_name"
   _full_name=${_full_branch_name#remotes\/origin\/*}
   echo git checkout $_full_name
   git checkout $_full_name
   echo git pull
   git pull
   echo cd $_therockorig
   cd $_therockorig
   echo git submodule set-branch -b $_full_name $_dir 
   git submodule set-branch -b $_full_name $_dir
   echo git add $_dir --sparse
   git add $_dir --sparse
   echo "====== DONE WITH $_reponame"
done

cd $_curdir
