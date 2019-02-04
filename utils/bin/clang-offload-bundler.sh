#!/bin/bash
#
#  $AOMP/bin/clang-offload-bundler.sh
#
#  This is a temporary bash script to demonstrate how we 
#  want to handle archive libraries as inputs to clang.
#  Right now, clang treats the .a as an object so it 
#  calls clang-offload-bundler to split the object into
#  its host and device components. 
#
#  The long term solution is to teach clang about .a files
#  and create an action to unarchive and unbundle the components
#  of the archive library to pass to respective gpu and host 
#  link steps. 
#  
#  This is the logic of this script:
#  If input to the unbundler is an archive file take these steps:
#    - unarchive the archive
#    - unbundle each object in the archive
#    - ld the host object files into a single .o file 
#    If device objects are cubin 
#      nvlink the device object files into a single cubin
#   else 
#      llvm-link the device object files into a single llvm ir bc file
#  Else 
#    - Call clang-offload-bundler with same arguments.
#
# 
inputfile=`echo "$3" | cut -d"=" -f2`
inputfiletype=`file $inputfile | cut -d":" -f2`
AOMP=${AOMP:-$HOME/rocm/aomp}
if [ ! -d $AOMP ] ; then
   AOMP="/opt/rocm/aomp"
fi
if [ ! -d $AOMP ] ; then
   echo "ERROR: AOMP not found at $AOMP"
   echo "       Please install AOMP or correctly set env-var AOMP"
   exit 1
fi
CUDA="${CUDA:-/usr/local/cuda}"

if [ "$inputfiletype" == " current ar archive" ] ; then
   ofile=`echo "$4" | cut -d"=" -f2`
   hostfile=`echo "$ofile" | cut -d"," -f1`
   devicefile=`echo "$ofile" | cut -d"," -f2`
   tmpdir=/tmp/unbundle$$
   curdir=$PWD
   mkdir -p $tmpdir
   cd $tmpdir
   $AOMP/bin/llvm-ar x $curdir/$inputfile
   hostfiles=""
   devicefiles=""
   for f in `$AOMP/bin/llvm-ar t $curdir/$inputfile` ; do
      $AOMP/bin/clang-offload-bundler $1 $2 -inputs=$f -outputs=host.$f,$f.bc $5
      rm $f
      devicefiles="$devicefiles $f.bc"
      hostfiles="$hostfiles host.$f"
      devicefiletype=`file $f.bc | cut -d":" -f2 | cut -d" " -f2`
   done
   #  Host files are in standard object format 
   ld -r  $hostfiles -o $hostfile
   #  device files could be cubin or llvm ir
   if [ "$devicefiletype" == "ELF" ] ; then
     # parse the devicefilename for the subarchitecture
     gpuarch=`echo $devicefile | cut -d"-" -f2`
     # FIXME:, for some reason this devicefile is not linking with the master
     #         and symbols are missing at runtime.
     $CUDA/bin/nvlink -v -o $devicefile -arch $gpuarch $devicefiles
   else
     #  amdgcn objects are LLVM IR , so we only need to llvm-link them
     $AOMP/bin/llvm-link $devicefiles -o $devicefile
   fi
   rm -rf $tmpdir
else
  # not an archive, use the original clang-offload-bundler
  $AOMP/bin/clang-offload-bundler $@
fi
