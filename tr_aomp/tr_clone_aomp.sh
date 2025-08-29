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
#
function test_apply_patch() {
   if ! patch -p1 -t -N --dry-run < $_patch_file  >/dev/null; then
      echo "ERROR:  patch --dry-run failed.  Could not apply $_patch_file "
      $thisdir/tr_add_info.sh patch_applied FAILED
      cd $_curdir
      exit 1
   else
      echo "patch -p1 --no-backup-if-mismatch < $_patch_file"
      patch -p1 --no-backup-if-mismatch < $_patch_file 
      $thisdir/tr_add_info.sh patch_applied YES
   fi
}

_therockdir=$TR_AOMP_REPOS/TheRock
_curdir=$PWD

# When AOMP release info file does NOT exist, we initialize a new release with
# the tip of TheRock. This script will create the info file for subsequent
# executions of tr_clone_aomp.sh. AOMP_INIT_THEROCK_TIP is only used in this scrpt. 
# This is typically only done by the AOMP release manager. 
if [ -f $AOMP_INFO_FILE ] ; then
   AOMP_INIT_THEROCK_TIP=0
   _releaseshakey=`grep "^therock_shakey:" $AOMP_INFO_FILE | cut -d":" -f2- | xargs`
else
   echo
   echo "===== WARNING: Starting the new AOMP release $AOMP_VERSION_STRING."
   echo "      It is assumed you are the AOMP release manager or you just want to build"
   echo "      the tip of TheRock using the amd-staging branch of amd-llvm and hipify."
   echo 
   AOMP_INIT_THEROCK_TIP=1
   mkdir -p $AOMP_VERSION_DIR
fi

# tr_aomp_common_vars ensures that TR_AOMP_REPOS is created
mkdir -p $TR_AOMP_REPOS
if [ ! -d $TR_AOMP_REPOS ] ; then 
   echo "ERROR: $0 could not create directory $TR_AOMP_REPOS"
   exit 1
fi

cd $TR_AOMP_REPOS
echo
echo "===== Cloning or updating aomp repo"
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

if [ -d $_therockdir ] ; then 
   echo
   echo "===== $_therockdir already exists, so not cloning TheRock."
   _new_rock_repo=0
else
   cd $TR_AOMP_REPOS
   echo
   echo "===== git clone https://github.com/ROCm/TheRock.git -b main TheRock"
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

cd $_therockdir
if [ -d $_therockdir/.venv/bin ] ; then
   export PATH=$_therockdir/.venv/bin:$PATH
else
   echo "WARNING: .venv/bin directory is missing"
fi

if [ $AOMP_INIT_THEROCK_TIP == 1 ] ; then
   if [ $_new_rock_repo == 0 ] ; then
      echo
      echo "===== Removing residual updates to support checking out TheRock main"
      echo "      checking out main could also wipeout changes to amd-staging branches."
      _tmpfile=/tmp/submod$$
      git submodule > $_tmpfile
      while read _line ; do
         _subdir=`echo $_line | cut -d" " -f2`
         cd $_therockdir/$_subdir
         if [ "$_subdir" != "compiler/amd-llvm" ] && [ "$_subdir" != "compiler/hipify" ] ; then
            git checkout .
         fi
      done < $_tmpfile
      rm $_tmpfile
      cd $_therockdir
      echo git checkout .
      git checkout .
   fi
   
   echo "git checkout main"
   git checkout main
   if [ $? != 0 ] ; then
      echo "ERROR: Could not checkout main"
      cd $_curdir
      exit 1
   fi
   echo git pull
   git pull

   echo
   echo "===== Creating new $AOMP_INFO_FILE"
   # Save the main (tip) shakey that identifies this AOMP release.
   _releaseshakey=`git log -1 | grep commit | cut -d" " -f2`
   $thisdir/tr_add_info.sh therock_shakey $_releaseshakey
   _date=`date`
   $thisdir/tr_add_info.sh start_date $_date
   $thisdir/tr_add_info.sh aomp_version $AOMP_VERSION_STRING
   # Make initial patch empty.
   mkdir $AOMP_PATCH_DIR
   $thisdir/tr_add_info.sh user $USER
   _hostname=`hostname`
   $thisdir/tr_add_info.sh hostname $_hostname
   $thisdir/tr_add_info.sh amd-llvm_branch amd-staging
   $thisdir/tr_add_info.sh hipify_branch amd-staging
   $thisdir/tr_add_info.sh sources_fetched FALSE
fi

_first_time_on_an_initialized_release=0
if [ $_new_rock_repo == 1 ] ; then
   echo
   echo "===== Now using frozen shakey $_releaseshakey for AOMP $AOMP_VERSION_STRING"
   git checkout $_releaseshakey
   $thisdir/tr_add_info.sh sources_fetched FALSE
else
   _current_shakey=`git log -1 | grep commit | cut -d" " -f2`
   if [ $_releaseshakey != $_current_shakey ] ; then
      # THIS IS FIRST TIME ON NEW BUT INITIALIZED RELEASE
      # THis is first time on new release that has been initialized. 
      # So we must reset from _current_shakey to _releaseshakey
      echo
      echo "===== WARNING: Your current TheRock repo is at shakey $_current_shakey but "
      echo "      $AOMP_INFO_FILE requires $_releaseshakey"
      echo "      Assuming this is first time on an already initialized new release"
      echo "      Removing all residual changes from last release ... "
      echo
      _tmpfile=/tmp/submod$$
      git submodule > $_tmpfile
      while read _line ; do
         _subdir=`echo $_line | cut -d" " -f2`
         if [ "$_subdir" != "compiler/amd-llvm" ] && [ "$_subdir" != "compiler/hipify" ] ; then
            echo cd $_therockdir/$_subdir
            cd $_therockdir/$_subdir
            echo git checkout .
            git checkout .
         fi
      done < $_tmpfile
      rm $_tmpfile
      echo cd $_therockdir
      cd $_therockdir
      echo git checkout .
      git checkout .
      git " Running: git checkout $_releaseshakey"
      git checkout $_releaseshakey
      _first_time_on_an_initialized_release=1
      $thisdir/tr_add_info.sh sources_fetched FALSE
   fi
fi

_sources_fetched=`grep "^sources_fetched:" $AOMP_INFO_FILE | cut -d":" -f2- | xargs`
if [ "$_sources_fetched" != "TRUE" ] ; then 
   echo
   echo "=====  _sources_fetched=$_sources_fetched. Running python ./build_tools/fetch_sources.py"
   python ./build_tools/fetch_sources.py
   echo "=====  Done running python ./build_tools/fetch_sources.py"
   $thisdir/tr_add_info.sh sources_fetched TRUE
else
   echo
   echo "=====  _sources_fetched=$_sources_fetched. Sources have already been fetched"
fi

# AOMP needs amd-staging branches of amd-llvm and hipify
echo 
echo "===== Checking out amd-staging for amd-llvm and hipify"
cd $TR_AOMP_REPOS/TheRock/compiler/amd-llvm
echo git checkout amd-staging
git checkout amd-staging
echo git pull
git pull
cd $TR_AOMP_REPOS/TheRock/compiler/hipify
echo git checkout amd-staging
git checkout amd-staging
echo git pull
git pull
echo "===== DONE checking out amd-staging for amd-llvm and hipify ===== "

_do_aomp_patches="$(( $_first_time_on_an_initialized_release == 1 || $(( $_new_rock_repo == 1 && $AOMP_INIT_THEROCK_TIP == 0 )) ))"
if [[ $_do_aomp_patches == 1 ]] ; then 
   echo
   echo "===== Attempting to apply patches in $AOMP_PATCH_DIR ====="
   $thisdir/tr_add_info.sh patch_dir $AOMP_PATCH_DIR
   echo cd $_therockdir
   cd $_therockdir
   _patch_file=$AOMP_PATCH_DIR/_TheRock.patch
   test_apply_patch
   _tmpfile=/tmp/submod$$
   git submodule > $_tmpfile
   while read _line ; do
      _subdir=`echo $_line | cut -d" " -f2`
      cd $_therockdir/$_subdir
      _subdirname=`echo $_subdir | tr "/" "_"`
      _patch_file=$AOMP_PATCH_DIR/$_subdirname.patch
      test_apply_patch
   done < $_tmpfile
   rm $_tmpfile
else
   echo
   echo "===== Skipping patch $AOMP_PATCH_DIR ====="
fi

if [ $AOMP_INIT_THEROCK_TIP == 1 ] ; then
   cd $_therockdir
   git submodule >$AOMP_SUBMODS_FILE
   echo
   echo "=====  Initialization of new AOMP release $AOMP_VERSION_STRING COMPLETE!"
   echo "       No AOMP patch has been applied. "
   echo "       Please consider doing the following steps NOW:"
   echo "  1. Try apply last AOMP release patch to TheRock repo."
   echo "  2. Try TheRock patches left behind at $_therockdir/patches/amd-mainline/llvm-project"
   echo "     It would be better if these patches were made upstream or merged in amd-staging."
   echo "  3. correct issues by testing build and updating TheRock or its submodles. DO NOT"
   echo "     CHECK ANYTHING IN because tr_create_patch.sh uses git diff"
   echo "  4. Only after last successful build, create new patch with tr_create_patch.sh"
   echo "     and review files in $AOMP_PATCH_DIR"
   echo "  5. Review $AOMP_INFO_FILE"
   echo "  6. To start team development of $AOMP_VERSION_STRING add, commit,"
   echo "     and push the following to start development"
   echo "       $TR_AOMP_REPOS/aomp/tr_aomp/tr_aomp_common_vars"
   echo "       $AOMP_INFO_FILE"
   echo "       All files in $AOMP_PATCH_DIR"
   echo "       $AOMP_SUBMODS_FILE"
   echo "       all files in $AOMP_BUILD_LOGS"
   echo "  7. Email/chat team to pull updates, run tr_clone_aomp.sh and tr_build_aomp.sh"
   echo "     Any existing local changes to llvm-project or hipify will not be lost"
   echo
fi

# Create convenience link for developers
cd $TR_AOMP_REPOS
if [ ! -L llvm-project ] ; then
   ln -sr TheRock/compiler/amd-llvm llvm-project
fi

cd $_curdir
echo
echo "===== DONE $0 for AOMP release $AOMP_VERSION_STRING"
