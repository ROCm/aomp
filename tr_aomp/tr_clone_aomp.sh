#!/bin/bash
#
#  tr_clone_aomp.sh: Clone TheRock repository to use to build aomp.sh
#                    using TheRock repo and its submodules. unlike clone_aomp.sh
#                    this script is NOT (yet) reusable to refresh all the repos. 
#
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/tr_aomp_common_vars"
# --- end standard header ----

_rockorigdir=$TR_AOMP_REPOS/TheRock.orig
_therockdir=$TR_AOMP_REPOS/TheRock
_curdir=$PWD

# tr_aomp_common_vars ensures that TR_AOMP_REPOS is set
mkdir -p $TR_AOMP_REPOS
if [ ! -d $TR_AOMP_REPOS ] ; then 
   echo "ERROR: $0 could not create directory $TR_AOMP_REPOS"
   exit 
fi

cd $TR_AOMP_REPOS
if [ -d $TR_AOMP_REPOS/aomp ] ; then 
   echo "WARNING: Skipping clone of aomp, $TR_AOMP_REPOS/aomp already exists"
else
   echo git clone -b aomp-dev https://github.com/ROCm/aomp
   git clone -b aomp-dev https://github.com/ROCm/aomp
fi
echo cd $TR_AOMP_REPOS/aomp
cd $TR_AOMP_REPOS/aomp
echo git checkout aomp-dev
git checkout aomp-dev
echo git pull
git pull
echo "==== DONE cloning or updating aomp repo ===="
echo

if [ -d $_therockdir ] ; then 
   echo "$_therockdir already exists, so only pulling updates to amd-staging submodules"
   # FIXME , ensure on saved hash here
   _new_rock_repo=0
else
   cd $TR_AOMP_REPOS
   echo git clone https://github.com/ROCm/TheRock.git -b main TheRock
   git clone https://github.com/ROCm/TheRock.git -b main TheRock
   cd TheRock
   echo git submodule init
   git submodule init
   echo git submodule update
   git submodule update
   # Initialization the environment without fetch_sources.py
   echo "python3 -m venv .venv && source .venv/bin/activate"
   python3 -m venv .venv && source .venv/bin/activate
   echo "pip install -r requirements.txt"
   pip install -r requirements.txt
   _new_rock_repo=1
fi

echo cd $_therockdir
if [ -d $_therockdir/.venv/bin ] ; then
   echo "adding $_therockdir/.venv/bin to PATH"
   export PATH=$PWD/.venv/bin:$PATH
   which python
else
   echo "WARNING: .venv/bin directory is missing"
   which python
fi

if [ $_new_rock_repo == 1 ] ; then
   if [ $AOMP_BUILD_FROZEN_ROCK == 0 ] ; then
      echo "WARNING: AOMP_BUILD_FROZEN_ROCK=0 is for starting new AOMP release."
      echo git checkout main
      git checkout main
      echo git pull
      git pull
      echo "--- git status from branch main with AOMP_BUILD_FROZEN_ROCK=0"
      git status
      echo "--- git status end"
      echo
      # Save the main (tip) shakey that identifies this AOMP release.
      _shakey=`git log -1 | grep commit | cut -d" " -f2`
      echo "$thisdir/tr_add_info.sh therock_shakey $_shakey"
      $thisdir/tr_add_info.sh therock_shakey $_shakey
      _date=`date`
      $thisdir/tr_add_info.sh start_date $_date
      $thisdir/tr_add_info.sh aomp_version $AOMP_VERSION_STRING
      $thisdir/tr_add_info.sh patch_file patches/tr_aomp_/$AOMP_VERSION_STRING.patch
      $thisdir/tr_add_info.sh user $USER
      _hostname=`hostname`
      $thisdir/tr_add_info.sh hostname $_hostname
      $thisdir/tr_add_info.sh staging_repos llvm-project hipify
   else
      _shakey=`grep "^therock_shakey:" $AOMP_INFO_FILE | cut -d":" -f2- | xargs`
      echo "using default frozen rock shakey $_shakey"
      echo git checkout $_shakey
      git checkout $_shakey
      echo "--- git status for TheRock following shakey checkout from $AOMP_INFO_FILE"
      git status
      echo "--- git status end"
   fi
   cd $_therockdir
   echo "===== IN $PWD ====> running python ./build_tools/fetch_sources.py"
   python ./build_tools/fetch_sources.py
fi

# Regardless of new or frozen TheRock, AOMP needs lastest amd-staging branch of
# both llvm-project and hipify  amd-staging branch.
echo 
echo "====== checking out amd-staging for amd-llvm and hipify"
cd $TR_AOMP_REPOS/TheRock/compiler/amd-llvm
git checkout amd-staging
git pull
cd $TR_AOMP_REPOS/TheRock/compiler/hipify
git checkout amd-staging
git pull

if [ $_new_rock_repo == 1 ] ; then
   # save the current state of each submodule and parent to be used
   # when creating patch. See tr_create_patch_from_orig.sh
   echo cd $_therockdir
   cd $_therockdir
   git submodule > $_tmpfile
   while read _line ; do
      #echo "LINE=$_line"
      _subdir=`echo $_line | cut -d" " -f2`
      cd $_therockdir/$_subdir
      if [ "$_subdir" != "compiler/amd-llvm" ] && [ "$_subdir" != "compiler/hipify" ] ; then
         echo "DIR:$PWD "
         echo git switch -c tr_aomp_orig_$AOMP_VERSION_STRING
         git switch -c tr_aomp_orig_$AOMP_VERSION_STRING
         echo git add -A
         git add -A
         echo "git commit -m Creation of branch tr_aomp_orig_$AOMP_VERSION_STRING"
         git commit -m "Creation of branch tr_aomp_orig_$AOMP_VERSION_STRING"
         echo git switch - --detach
         git switch - --detach
      fi
   done < $_tmpfile
   rm $_tmpfile
   cd $_therockdir
   echo git switch -c tr_aomp_orig_$AOMP_VERSION_STRING
   git switch -c tr_aomp_orig_$AOMP_VERSION_STRING
   echo git add -A
   git add -A
   echo "git commit -m Creation of branch tr_aomp_orig_$AOMP_VERSION_STRING"
   git commit -m "Creation of branch tr_aomp_orig_$AOMP_VERSION_STRING"
   echo git switch - --detach
   git switch - --detach

   if [ $AOMP_BUILD_FROZEN_ROCK == 1 ] ; then
      echo
      echo "========= Applying tr_aomp_$AOMP_VERSION_STRING.patch ==========="
      echo cd $TR_AOMP_REPOS/TheRock
      cd $TR_AOMP_REPOS/TheRock
      echo "patch -p1 < $TR_AOMP_REPOS/aomp/tr_aomp/patches/tr_aomp_$AOMP_VERSION_STRING.patch"
      patch -p1 < $TR_AOMP_REPOS/aomp/tr_aomp/patches/tr_aomp_$AOMP_VERSION_STRING.patch
      echo "--- git status for TheRock following patch $TR_AOMP_REPOS/aomp/tr_aomp/patches/tr_aomp_$AOMP_VERSION_STRING.patch"
      git status
      echo "--- git status end"
   else
      echo "WARNING: AOMP_BUILD_FROZEN_ROCK=0 is for starting new AOMP release."
      echo "         Apply old AOMP release patch, correct issues, then create new patch in:"
      echo "         $TR_AOMP_REPOS/aomp/tr_aomp/patches/tr_aomp_$AOMP_VERSION_STRING.patch"
   fi
fi

# Create convenience link for developers
cd $TR_AOMP_REPOS
if [ ! -L llvm-project ] ; then
   ln -sr TheRock/compiler/amd-llvm llvm-project
fi

cd $_curdir
echo "DONE $0"
