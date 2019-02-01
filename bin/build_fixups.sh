#!/bin/bash
#
#   build_fixups.sh : make some fixes to the installation.
#                     We eventually need to remove this hack.
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

# Temporarily remove debug libdevice and replace with release 
savedir=$PWD
echo cd $AOMP/lib-debug
cd $AOMP/lib-debug
if [ -L libdevice ] ; then
   $SUDO rm libdevice
fi
if [ -d libdevice ] ; then
  $SUDO rm -rf libdevice
fi
# Now link to nondefault libdevice
echo $SUDO ln -sf ../lib/libdevice libdevice
$SUDO ln -sf ../lib/libdevice libdevice
cd $savedir

# Copy examples 
if [ -d $AOMP/examples ] ; then 
  $SUDO rm -rf $AOMP/examples
fi
echo $SUDO cp -rp $AOMP_REPOS/$AOMP_REPO_NAME/examples $AOMP
$SUDO cp -rp $AOMP_REPOS/$AOMP_REPO_NAME/examples $AOMP

echo "Done with $0"
