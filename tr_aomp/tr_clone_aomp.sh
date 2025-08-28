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

_therockdir=$TR_AOMP_REPOS/TheRock
_curdir=$PWD

# When AOMP release info file does NOT exist, automatically initialize a new release
# with the tip of TheRock.  tr_clone_aomp.sh will create the info file for subsequent
# executions of tr_clone_aomp.sh. AOMP_INIT_THEROCK_TIP is only used in this scrpt. 
if [ -f $AOMP_INFO_FILE ] ; then
   AOMP_INIT_THEROCK_TIP=0
else
   AOMP_INIT_THEROCK_TIP=1
fi

# tr_aomp_common_vars ensures that TR_AOMP_REPOS is created
mkdir -p $TR_AOMP_REPOS
if [ ! -d $TR_AOMP_REPOS ] ; then 
   echo "ERROR: $0 could not create directory $TR_AOMP_REPOS"
   exit 
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
echo

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
   echo
   echo "WARNING: AOMP_INIT_THEROCK_TIP=1,  starting new AOMP release $AOMP_VERSION_STRING."
   echo "===== Removing residual updates to checkout TheRock main"
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
   echo git checkout main
   git checkout main
   if [ $? != 0 ] ; then
      echo "ERROR: Could not checkout main"
      exit 1
   fi
   echo git pull
   git pull

   echo
   echo "===== Creating aomp_$AOMP_VERSION_STRING.info"
   # Save the main (tip) shakey that identifies this AOMP release.
   _shakey=`git log -1 | grep commit | cut -d" " -f2`
   echo "$thisdir/tr_add_info.sh therock_shakey $_shakey"
   $thisdir/tr_add_info.sh therock_shakey $_shakey
   _date=`date`
   $thisdir/tr_add_info.sh start_date $_date
   $thisdir/tr_add_info.sh aomp_version $AOMP_VERSION_STRING
   # Make initial patch empty.
   echo touch $thisdir/patches/tr_aomp_/$AOMP_VERSION_STRING.patch
   touch $thisdir/patches/tr_aomp_$AOMP_VERSION_STRING.patch
   $thisdir/tr_add_info.sh patch_file patches/tr_aomp_$AOMP_VERSION_STRING.patch
   $thisdir/tr_add_info.sh user $USER
   _hostname=`hostname`
   $thisdir/tr_add_info.sh hostname $_hostname
   $thisdir/tr_add_info.sh amd-llvm_branch amd-staging
   $thisdir/tr_add_info.sh hipify_branch amd-staging
   echo cat aomp_$AOMP_VERSION_STRING.info
   cat aomp_$AOMP_VERSION_STRING.info
fi

echo
_shakey=`grep "^therock_shakey:" $AOMP_INFO_FILE | cut -d":" -f2- | xargs`
if [ $_new_rock_repo == 1 ] ; then
   echo
   echo "===== Now using frozen shakey $_shakey for AOMP $AOMP_VERSION_STRING"
   echo git checkout $_shakey
   git checkout $_shakey
else
   _current_shakey=`git log -1 | grep commit | cut -d" " -f2`
   if [ $_shakey != $_current_shakey ] ; then
      echo
      echo "===== WARNING: Your current TheRock repo is at shakey $_current_shakey but "
      echo "      $AOMP_INFO_FILE requires $_shakey"
      echo "      Running git checkout $_shakey"
      git checkout $_shakey
      echo "--- git status"
      git status
      echo "--- done git status"
      echo "NOTE: You may need to reapply patch tr_aomp_$AOMP_VERSION_STRING.patch"
   fi
fi

echo
echo "=====  running python ./build_tools/fetch_sources.py"
python ./build_tools/fetch_sources.py

# Regardless of tip or frozen shakey, AOMP needs specified branches of
# certain submodules, typically llvm-project(amd-llvm) and hipify.
# FIXME:  Getrepo_branch entries from info file.
echo 
echo "====== checking out amd-staging for amd-llvm and hipify"
cd $TR_AOMP_REPOS/TheRock/compiler/amd-llvm
echo git status for amd-llvm
git status
echo git checkout amd-staging
git checkout amd-staging
echo git pull
git pull

cd $TR_AOMP_REPOS/TheRock/compiler/hipify
echo git status for hipify
git status
echo git checkout amd-staging
git checkout amd-staging
echo git pull
git pull

if [ $_new_rock_repo == 1 ] || [ $AOMP_INIT_THEROCK_TIP == 1 ] ; then
   # save the current state of each submodule and parent to be used
   # when creating patch. See tr_create_patch_from_orig.sh
   echo
   echo "===== Creating original branches of each submodule to support patch creation"
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
fi

if [ $_new_rock_repo == 1 ] && [ $AOMP_INIT_THEROCK_TIP == 0 ] ; then
   echo
   echo "===== Applying tr_aomp_$AOMP_VERSION_STRING.patch ====="
   echo cd $TR_AOMP_REPOS/TheRock
   cd $TR_AOMP_REPOS/TheRock
   echo "patch -p1 < $TR_AOMP_REPOS/aomp/tr_aomp/patches/tr_aomp_$AOMP_VERSION_STRING.patch"
   patch -p1 < $TR_AOMP_REPOS/aomp/tr_aomp/patches/tr_aomp_$AOMP_VERSION_STRING.patch
fi

if [ $AOMP_INIT_THEROCK_TIP == 1 ] ; then
   echo
   echo "=====  AOMP_INIT_THEROCK_TIP=1 started the new AOMP release $AOMP_VERSION_STRING"
   echo "       No AOMP patch applied. Please consider the following steps NOW:"
   echo "  1. Try apply last AOMP release patch to TheRock repo."
   echo "  2. Try TheRock patches left behind at $_therockdir/patches/amd-mainline/llvm-project"
   echo "  3. correct issues by testing build and updating TheRock or its submodles"
   echo "  4. Create new patch with tr_create_patch_from_orig.sh"
   echo "     and review it, see patches/tr_aomp.patch"
   echo "  5. Review aomp_$AOMP_VERSION_STRING.info"
   echo "  6. copy tr_aomp.patch to tr_aomp_$AOMP_VERSION_STRING.patch"
   echo "  7. add, commit and push 3 files: tr_aomp_common_vars,"
   echo "     aomp_$AOMP_VERSION_STRING.info,  and tr_aomp_$AOMP_VERSION_STRING.patch "
   echo
fi

# Create convenience link for developers
cd $TR_AOMP_REPOS
if [ ! -L llvm-project ] ; then
   ln -sr TheRock/compiler/amd-llvm llvm-project
fi

cd $_curdir
echo "DONE $0"
