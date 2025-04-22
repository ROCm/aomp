#!/bin/bash
# Copyright(C) 2019 Advanced Micro Devices, Inc. All rights reserved.
#
# unbundle.sh: use clang clang-offload-bundler tool to unbundle files
#              This script is the companion to bundle.sh.  It will name the 
#              generated files using the conventions used in the coral compiler. 
#
#  Written by Greg Rodgers
# 
AOMP=${AOMP:-$HOME/rocm/aomp}
if [ ! -d $AOMP ] ; then
   AOMP="/opt/rocm/aomp"
fi
if [ ! -d $AOMP ] ; then
   echo "ERROR: AOMP not found at $AOMP"
   echo "       Please install AOMP or correctly set env-var AOMP"
   exit 1
fi
EXECBIN=$AOMP/bin/clang-offload-bundler
if [ ! -f $EXECBIN ] ; then 
   echo "ERROR: $EXECBIN not found"
   exit 1
fi

infilename=${1:-ll}
if [ -f $infilename ] ; then 
   ftype=${infilename##*.}
   tnames="$infilename"
else
#  Input was not a file, work on all files in the directory of specified type
   ftype=$infilename
   tnames=`ls *.$ftype 2>/dev/null`
   if [ $? != 0 ] ; then 
      echo "ERROR: No files of type $ftype found"
      echo "       Try a filetype other than $ftype"
      exit 1
   fi
fi

ftypeout=$ftype

for tname in $tnames ; do 
   mname=${tname%.$ftype}
   if [ "$ftype" == "o" ] ; then 
      otargets=`strings $tname | grep "openmp-"`
      for longtarget in $otargets ; do
         if [ "${longtarget:0:22}" == "__CLANG_OFFLOAD_BUNDLE" ] ; then
            targets="$targets ${longtarget:24}"
         else
            targets="$targets $longtarget"
         fi
      done
      host=`strings $tname | grep "host-"`
      if [ "${longtarget:0:22}" == "__CLANG_OFFLOAD_BUNDLE" ] ; then
         host=${host:24}
      else
         host=${host%*BC}
      fi
      targets="$targets $host"
   else
      if [ "$ftype" != "bc" ] ; then 
         targets=`grep "__CLANG_OFFLOAD_BUNDLE____START__" $tname | cut -d" " -f3`
      else
#        Hack to find bc targets from a bundle
         for t in `strings $tname | grep "openmp-" ` ; do 
            t=${t%*k}
            t=${t%*\'>}
            targets="$targets $t"
         done 
         host=`strings $tname | grep "host-" `
         targets="$targets ${host%*BC}"
      fi
   fi
   targetlist=""
   sepchar=""
   fnamelist=""
   for target in $targets ; do 
      targetlist=$targetlist$sepchar$target
      if [ "$ftype" == "o" ] ; then
         if [ "${target:0:6}" == "openmp" ] && [ "${target:7:6}" == "amdgcn" ] ; then
            ftypeout="bc"
         fi
      fi
      fnamelist=$fnamelist$sepchar${tname}.$target
      sepchar=","
   done
   if [ "$targetlist" != "" ] ; then 
      cmd="$EXECBIN -unbundle -type=$ftypeout -inputs=$tname -targets=$targetlist -outputs=$fnamelist"
      echo $cmd
      $cmd
      if [ $? != 0 ] ; then 
         echo "ERROR: $EXECBIN failed."
         echo "       The failed command was:"
         echo $cmd
         exit 1
      fi
   fi
done
