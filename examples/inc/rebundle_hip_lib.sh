#!/bin/bash
#
# rebundle_hip_lib.sh : Utility that takes a static device library
#                       built with packager and converts it to 
#                       a static device library built with bundler.
#

#!/bin/bash
#   rebundle_hip_lib.sh: Utility to convert a static device library (SDL) built
#         with clang-offload-packager into a static device library for hip. HIP 
#         currently expects SDLs to be built with clang-offload-bundler. 
#
# Copyright(C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
PROGVERSION=X.Y-Z
function version(){
   echo $0 version $PROGVERSION
   exit 0
}
function usage(){
/bin/cat 2>&1 <<"EOF"

   rebundle_hip_lib.sh: Utility to convert a static device library (SDL) built
         with clang-offload-packager into a static device library for hip. HIP
         currently expects SDLs to be built with clang-offload-bundler.  This
         utility is currently necessary if an SDL is built ith OpenMP target
         offload and user wants to call the target functions from a HIP kernel.

   Usage:
      rebundle_hip_lib.sh lib_ompSDL.a                   Convert lib_ompSDL.a to
                                                            bundled/lib_ompSDL.a

      rebundle_hip_lib.sh lib_ompSDL.a bundled/lib_ompSDL.a       Same as above.

      hipcc ...  -L($PWD/bundled) -l _ompSDL         Use converted SDL with hip.

   Options:
      -v   Print verbose messages
      -h   Print this help message and exit
      --version Print version and exit

   Environment variables:
      LLVM_INSTALL_DIR    LLVM installation directory, defaultis /opt/rocm/llvm

   Copyright (c) 2024  ADVANCED MICRO DEVICES, INC.

EOF
  exit 0
}

# SEt LLVM_INSTALL_DIR if it is not already set
LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR:-/opt/rocm/llvm}

case "$1" in
   -v)          _VERBOSE=true ; shift;;
   -h)          usage ;;
   -help)       usage ;;
   --help)      usage ;;
   -version)    version ;;
   --version)   version ;;
esac

if [ -z $1 ] ; then 
  echo "ERROR: $0 requires at least one input argument."
  exit 1
fi
if [ ! -f $1 ] ; then 
  echo "ERROR: INPUT FILE $1 not found"
  exit 1
fi
_packaged_archive=${1}
_packaged_archive_dir=`dirname $_packaged_archive`
[ "$_packaged_archive_dir" == "." ] && _packaged_archive_dir=$PWD
_packaged_archive_file=$(basename ${_packaged_archive})
_new_bundled_archive=${2:-$_packaged_archive_dir/bundled/$_packaged_archive_file}
_new_bundled_archive_dir=`dirname $_new_bundled_archive`
mkdir -p $_new_bundled_archive_dir

if [ -f $_new_bundled_archive ] ; then 
  echo "WARNING: OUTPUT FILE $_new_bundled_archive exists and will be overwritten"
  if [ $_new_bundled_archive == $_packaged_archive ] ; then 
     echo "ERROR: Input packaged_archive $_packaged_archive cannot be same file as OUTPUT _new_bundled_archive"
     exit 1
  fi
fi

curdir=$PWD
tdir=/tmp/$USER/rebundle$$
[ $_VERBOSE ] && echo "mkdir -p $tdir/bundled"
mkdir -p $tdir/bundled
[ $_VERBOSE ] && echo "cd $tdir"
cd $tdir


num_objs=0
obj_files=()
cmd="$LLVM_INSTALL_DIR/bin/llvm-ar t $_packaged_archive_dir/$_packaged_archive_file"
$cmd > _tfile_list_of_objfiles
while read -r objfile; do 
  obj_files+=$objfile
  num_objs=$(( $num_objs + 1 ))
done < _tfile_list_of_objfiles

cmd="$LLVM_INSTALL_DIR/bin/llvm-ar x $_packaged_archive_dir/$_packaged_archive_file"
$cmd

bundled_objects=""

for _objfnum in `seq 0 $(( $num_objs- 1 ))` ; do 
  #echo "-------------- objfile[$_objfnum]= ${obj_files[$_objfnum]} num_objs=$num_objs"
  objfile=${obj_files[$_objfnum]} 
  $LLVM_INSTALL_DIR/bin/llvm-objdump $objfile --offloading >tfile_objdump.stdout
  grep IMAGE tfile_objdump.stdout > tfile_images
  if [ $? != 0 ] ; then
     echo "WARNING: member $objfile of $_packaged_archive has no packaged images"
     echo "         Check that $_packaged_archive is correct"
  fi
  num_images=0
  image_arch=()
  image_triple=()
  image_producer=()
  while read -r line ; do 
    grep -A4 ""$line"" tfile_objdump.stdout >tfile_image_fields 2>/dev/null
    image_arch+=`grep arch tfile_image_fields | awk '{print $2}'` 
    image_triple+=`grep triple tfile_image_fields | awk '{print $2}'` 
    image_producer+=`grep producer tfile_image_fields | awk '{print $2}'` 
    num_images=$(( $num_images + 1 ))
  done <tfile_images

  bundled_targets=""
  bundled_inputs=""
  for _i in `seq 0 $(( $num_images - 1 ))` ; do 
    _extracted_file="file${_objfnum}_image_${_i}_unpackaged.bc"
    cmd="${LLVM_INSTALL_DIR}/bin/clang-offload-packager $objfile --image=file=${_extracted_file},triple=${image_triple[$_i]},arch=${image_arch[$_i]},kind=${image_producer[$_i]}"
    #echo $cmd
    [ $_VERBOSE ] && echo $cmd
    $cmd
    bundled_targets+="${image_producer[$_i]}-${image_triple[$_i]}-${image_arch[$_i]},"
    bundled_inputs+="-input=${_extracted_file} "
  done
  bundled_targets+="host-x86_64-unknown-linux-gnu"
  bundled_inputs+=" -input=$objfile"
  bcmd="${LLVM_INSTALL_DIR}/bin/clang-offload-bundler --type=o --targets=$bundled_targets --output=bundled/$objfile $bundled_inputs"
  bundled_objects+=" bundled/$objfile"
  [ $_VERBOSE ] && echo $bcmd
  $bcmd
done 

cmd="$LLVM_INSTALL_DIR/bin/llvm-ar rcs $_new_bundled_archive $bundled_objects"
[ $_VERBOSE ] && echo $cmd
$cmd

cd $curdir
rm -rf $tdir
[ $_VERBOSE ] && echo rm -rf $tdir
rm -rf $tdir
