#!/bin/bash
#
#  tr_list_submodules.sh
#     
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/tr_aomp_common_vars"
# --- end standard header ----
 
_therockdir=$TR_AOMP_REPOS/TheRock
_curdir=$PWD

cd $_therockdir
if [ ! -d .git ] ; then 
   echo "ERROR: Directory $_therockdir is not a git clone." >&2
   echo "       Run $0 from TheRock root directory." >&2
   exit 1
fi

# Collect info from aomp manifest which is read by tr_list_aomp_branches.sh
_aomp_repos_temp=()
for _line in `$TR_AOMP_REPOS/aomp/tr_aomp/tr_list_aomp_branches.sh | awk '{print $2 ":" $4}' | tr -d '"' ` ; do 
   _aomp_repos_temp+=($_line)
done
# Fix certain aomp reponames
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
   cd $_therockdir
   if [ -d $_category ] ; then 
      _dir=$_line
      _reponame=${_dir##*/}
      if [ "$_reponame" == "binaries" ] ; then 
        _reponame=${_dir#*/}
      fi
      # search for the branch we want for this submodule
      cd $_therockdir/$_dir
      _aomp_branch_name=""
      for _repo in ${aomp_repos[@]} ; do
         _aomp_reponame=${_repo#*:}
	 if [ $_reponame == $_aomp_reponame ] ; then 
           _aomp_branch_name=${_repo%:*}
	 fi
      done
      if [ -z "$_aomp_branch_name" ] ; then 
         echo "WARNING: NO AOMP REPO FOR SUBMODULE:$_reponame $_dir" >&2
      else
         submodules+=($_reponame:$_category:$_dir:$_aomp_branch_name)
      fi
   fi
done 

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
      echo "WARNING: NO SUBMODULE FOR AOMP REPO $_aomp_reponame" >&2
   fi
done

joined=()
for _submodule in ${submodules[@]} ; do
   _dir=`echo $_submodule | cut -d":" -f3`
   _aomp_branch_name=`echo $_submodule | cut -d":" -f4`
   cd $_therockdir/$_dir
   if [ -f .git ] ; then 
      _mod_dir=`cat .git | cut -d":" -f2`
      _therockdir_branch_name=`cat $_mod_dir/HEAD | cut -d":" -f2 | awk '{print $1}'`
      _therockdir_branch_name=${_therockdir_branch_name#refs\/heads\/*}
      _entry_colons="$_dir":"$_therockdir_branch_name":"$_aomp_branch_name"
      joined+=("$_entry_colons")
   else
      echo "WARNING: SUBMODULE $_dir DOES NOT HAVE A .git FILE AT $_therockdir/$_dir/.git" >&2
   fi
done

for _join in ${joined[@]} ; do
   _dir=`echo $_join | cut -d":" -f1`
   _therock_branch_name=`echo $_join | cut -d":" -f2`
   _aomp_branch_name=`echo $_join | cut -d":" -f3`
   if [ "$_aomp_branch_name" == "$_therock_branch_name" ] ; then 
      printf "%40s %24s\n" "$_dir" "$_therock_branch_name"
   else
      echo "WARNING: $_dir reset branch $_therock_branch_name to $_aomp_branch_name" >&2
   fi
done
cd $_curdir
