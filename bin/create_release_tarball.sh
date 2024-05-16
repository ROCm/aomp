#!/bin/bash
# 
#  create_release_tarball.sh: 
#
#  This is how we create the release source tarball.  Only run this after a successful build
#  so that this captures patched files because the root Makefile turns off patching
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

function getmanifest(){
  if [[ "$AOMP_VERSION" == "13.1" ]] || [[ $AOMP_MAJOR_VERSION -gt 13 ]] ; then
    # For 13.1 and beyond, we use a manifest file to specify the repos to clone.
    # However, we gave up on using the repo command to clone the repos.
    # That is all done here by parsing the manifest file.
    if [ "$AOMP_MANIFEST_FILE"  != "" ]; then
      manifest_file=$AOMP_MANIFEST_FILE
    else
      ping -c 1 $AOMP_GIT_INTERNAL_IP 2> /dev/null >/dev/null
      if [ $? == 0 ] && [ "$AOMP_EXTERNAL_MANIFEST" != 1 ]; then
        # AMD internal repo file
        manifest_file=$thisdir/../manifests/aompi_${AOMP_VERSION}.xml
      else
        abranch=`git branch | awk '/\*/ { print $2; }'`
        # Use release manifest if on release branch
        if [ "$abranch" == "aomp-${AOMP_VERSION_STRING}" ]; then
          manifest_file=$thisdir/../manifests/aomp_${AOMP_VERSION_STRING}.xml
        else
          manifest_file=$thisdir/../manifests/aomp_${AOMP_VERSION}.xml
        fi
      fi
    fi
    if [ ! -f $manifest_file ] ; then
      echo "ERROR manifest file missing: $manifest_file"
      exit 1
    fi
     echo Using: $manifest_file
  else
    echo Error: This AOMP version does not have a manifest file.
  fi
}

function getreponame(){
  getmanifest
  tmpfile=/tmp/mlines$$
  tarballremove="roctracer rocprofiler aomp build Makefile"
  # Manifest file must be one project line per repo
  #manifest_file=/home/release/git/aomp14/aomp/manifests/aomp_14.0-0.xml
  cat $manifest_file | grep project > $tmpfile
  while read line ; do
    found=0
    for field in `echo $line` ; do
      if [ -z "${field##*path=*}" ]  ; then
        path=$(eval echo `echo $field | cut -d= -f2 `)
      fi
    done
  reponame=$path
  for component in $tarballremove; do
    if [ "$reponame" == "$component" ]; then
      found=1
      break
    fi
  done
  if [ "$found" == 0 ]; then
    repos="$repos $reponame"
  fi
  done <$tmpfile

  echo $repos
  rm $tmpfile
}

# Get repos from manifest
getreponame

REPO_NAMES=$repos
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
majorver=${AOMP_VERSION}
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
echo "------ DONE! CMD:$0  FILE:$tarball ------"
ls -lh $tarball
