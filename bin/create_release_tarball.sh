#!/bin/bash
# 
#  create_release_tarball.sh: 
#
#  This is how we create the release source tarball.  Only run this after a successful build
#  so that this captures patched files because the root Makefile turns off patching
#
# --- Start standard header to set build environment variables ----
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


REPO_NAMES="$AOMP_PROJECT_REPO_NAME $AOMP_LIBDEVICE_REPO_NAME $AOMP_VDI_REPO_NAME $AOMP_HIPVDI_REPO_NAME $AOMP_RINFO_REPO_NAME $AOMP_ROCT_REPO_NAME $AOMP_ROCR_REPO_NAME $AOMP_EXTRAS_REPO_NAME $AOMP_COMGR_REPO_NAME $AOMP_FLANG_REPO_NAME $AOMP_OCL_REPO_NAME $AOMP_DBGAPI_REPO_NAME $AOMP_GDB_REPO_NAME"
ALL_NAMES="$REPO_NAMES Makefile build aomp"
# Check for extra directories.  Note build is in the exclude list
for dir_name in `ls $AOMP_REPOS` ; do
   found=0
   for repo_name in $ALL_NAMES ; do
      if [ "$repo_name" == "$dir_name" ] ; then
         found=1
      fi
   done
   if [ $found == 0 ] ; then 
      echo
      echo "WARNING:  FILE OR DIRECTORY '$dir_name' IS NOT IN LIST OF REPOS TO tar"
      echo "          $REPO_NAMES"
      echo
      echo "          $dir_name WILL NOT BE ADDED TO SOURCE TARBALL."
      echo "          CHECK DIRECTORY $AOMP_REPOS ."
      echo "          HIT ENTER TO CONTINUE or CTRL-C TO CANCEL"
      read
   fi
done

patchloc=$thisdir/patches
export IFS=" "
echo "----------------- PRE-PATCH STATUS -----------------"
for repo_name in $REPO_NAMES ; do
   cd $AOMP_REPOS/$repo_name
   echo
   echo $repo_name: git status
   git status
done
echo "----------------- PATCHING REPOS -----------------"
for repo_name in $REPO_NAMES ; do
   echo
   echo patchrepo $AOMP_REPOS/$repo_name
   patchrepo $AOMP_REPOS/$repo_name
done
echo "----------------- POST-PATCH STATUS -----------------"
for repo_name in $REPO_NAMES ; do
   cd $AOMP_REPOS/$repo_name
   echo
   echo $repo_name: git status
   git status
done

# This file will be uploaded to the release directory
tarball="$AOMP_REPOS/../aomp-${AOMP_VERSION_STRING}.tar.gz"
tmpdir=/tmp/create_tarball$$
majorver=${AOMP_VERSION%%.*}
tardir=$tmpdir/aomp$majorver
echo "----- Building symbolic temp dir $tardir------------"
echo mkdir -p $tardir
mkdir -p $tardir
cd $tardir
#  Copy makefile to $tardir
echo cp -p $AOMP_REPOS/$AOMP_REPO_NAME/Makefile $tardir/Makefile
cp -p $AOMP_REPOS/aomp/Makefile $tardir/Makefile
for repo_name in $REPO_NAMES ; do
   echo ln -sf $AOMP_REPOS/$repo_name $repo_name
   ln -sf $AOMP_REPOS/$repo_name $repo_name
done
echo ln -sf $AOMP_REPOS/$AOMP_REPO_NAME $AOMP_REPO_NAME
ln -sf $AOMP_REPOS/$AOMP_REPO_NAME $AOMP_REPO_NAME
cd $tmpdir
cmd="tar --exclude-from $thisdir/create_release_tarball_excludes -h -czf $tarball aomp$majorver "
echo "----------------- START tar COMMAND -----------------"
echo time $cmd
time $cmd
echo 
echo done creating $PWD/$tarball
echo
echo "----- Cleanup symbolic temp dir $tardir------------"
echo cd $tardir
cd $tardir
echo "rm *"
rm *
echo rmdir $tardir
rmdir $tardir
cd /tmp
echo rmdir $tmpdir
rmdir $tmpdir

echo "----------------- REVERSE PATCHING -----------------"
for repo_name in $REPO_NAMES ; do
   removepatch $AOMP_REPOS/$repo_name
done

echo "----------------- POST REVERSE PATCH STATUS -----------------"
for repo_name in $REPO_NAMES ; do
   cd $AOMP_REPOS/$repo_name
   echo
   echo $repo_name: git status
   git status
done

echo 
cd $AOMP_REPOS/..
echo "------ DONE! CMD:$0  FILE:$PWD/$tarball ------"
ls -lh $tarball
