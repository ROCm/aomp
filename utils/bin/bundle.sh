#!/bin/bash
#
#  bundle.sh: Use the clang bundle tool to combine files for
#             different targets that were created by coral 3.8 compiler.  
#             This script encodes the conventions used by coral 3.8 compiler 
#             including the embedding of nvptx64 submodel by 
#             concatonating it to nvptx64 in the triple. e.g. nvptx64sm_35
#
#  Written by Greg Rodgers
#
#  Todo: 
#     Add -f option  to overwrite if outfile exists
#     Add -v option , remove all echos otherwise
#     Add -h option

AOMP=${AOMP:-$HOME/rocm/aomp}
if [ ! -d $AOMP ] ; then 
   AOMP="/opt/rocm/aomp"
fi
EXECBIN=$AOMP/bin/clang-offload-bundler
if [ ! -f $EXECBIN ] ; then 
   echo "ERROR: $EXECBIN not found "
   exit 1
fi

outfilename=${1}
if [ -f $outfilename ] ; then 
   echo "ERROR: File $outfilename exists, use -f to overwrite"
   exit 1
fi

ftype=${outfilename##*.}
tnames=`ls $outfilename.* 2>/dev/null`

fnamelist=""
targetlist=""
sepchar=""
hostfound=0
for tname in $tnames ; do 
   target=${tname#${outfilename}.}
   if [ "${target:0:5}" == "host-" ] || [ "${target:0:7}" == "openmp-" ] ; then  
      if [ "${target:0:5}" == "host-" ] ; then 
         hostfound=1
      fi 
      targetlist=$target${sepchar}$targetlist
      fnamelist=$tname$sepchar$fnamelist
      sepchar=","
   else
      echo 
      echo " WARNING:  The file $tname with target=$target will be ignored."
      echo 
   fi
done

if [ $hostfound == 0 ] ; then
    echo "ERROR: No host file found for $outfilename"
    exit 1
fi

if [ "$targetlist" != "" ] ; then
   echo $EXECBIN -type=$ftype -inputs=$fnamelist -targets=$targetlist -outputs=$outfilename
   $EXECBIN -type=$ftype -inputs=$fnamelist -targets=$targetlist -outputs=$outfilename
   if [ $? != 0 ] ; then 
      echo "ERROR: $EXECBIN failed."
      echo "       command was"
      echo $EXECBIN -type=$ftype -inputs=$fnamelist -targets=$targetlist -outputs=$outfilename
   fi
fi
