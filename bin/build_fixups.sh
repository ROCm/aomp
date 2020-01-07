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
if [[ "$AOMP_STANDALONE_BUILD" == 0 ]] ; then
  . $AOMP_REPOS/aomp/bin/aomp_common_vars
else
  [ ! -L "$0" ] || thisdir=$(getdname `readlink "$0"`)
  . $thisdir/aomp_common_vars
fi
# --- end standard header ----

# Copy examples 
if [ -d $AOMP/examples ] ; then 
  $SUDO rm -rf $AOMP/examples
fi
echo $SUDO cp -rp $AOMP_REPOS/$AOMP_REPO_NAME/examples $AOMP
$SUDO cp -rp $AOMP_REPOS/$AOMP_REPO_NAME/examples $AOMP

echo Cleaning AOMP Directory...
#examples
$SUDO rm -f $AOMP/examples/hip/*.txt
$SUDO rm -f $AOMP/examples/hip/*.sh
$SUDO rm -f $AOMP/examples/openmp/*.txt
$SUDO rm -f $AOMP/examples/openmp/*.sh
$SUDO rm -f $AOMP/examples/cloc/*.txt
$SUDO rm -f $AOMP/examples/cloc/*.sh
$SUDO rm -f $AOMP/examples/fortran/*.txt
$SUDO rm -f $AOMP/examples/fortran/*.sh
$SUDO rm -f $AOMP/examples/*.sh

#Clean libexec, share
$SUDO rm -rf $AOMP/libexec
$SUDO rm -rf $AOMP/share
#Clean hcc
$SUDO rm -rf $AOMP/hcc/bin
$SUDO rm -rf $AOMP/hcc/include
$SUDO rm -rf $AOMP/hcc/libexec
$SUDO rm -rf $AOMP/hcc/rocdl
$SUDO rm -rf $AOMP/hcc/share
#Clean hcc/lib
$SUDO rm -rf $AOMP/hcc/lib/clang
$SUDO rm -rf $AOMP/hcc/lib/cmake
$SUDO rm -rf $AOMP/hcc/lib/*.bc
$SUDO rm -rf $AOMP/hcc/lib/LLVM*
$SUDO rm -rf $AOMP/hcc/lib/libclang*
$SUDO rm -rf $AOMP/hcc/lib/libLLVM*
$SUDO rm -rf $AOMP/hcc/lib/libLTO*
$SUDO rm -rf $AOMP/hcc/lib/libRemarks*

echo "Done with $0"
