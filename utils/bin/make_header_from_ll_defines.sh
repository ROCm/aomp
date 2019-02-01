#!/bin/bash

# make_header_from_ll_defines.sh:
#    To use low level hc functions maintained in .ll, we need c/cpp headers so 
#    we can use them properly in hip device library implementations. 
#
# Usage: To build /tmp/hc_atomic_ll.h file run this command
#
#    make_header_from_ll_defines.sh $HOME/git/aomp/rocm-device-libs/hc/src/hc_atomic.ll
#    

llfile=${1:-/home/grodgers/git/aomp/rocm-device-libs/hc/src/hc_atomic.ll}
outfn=${llfile##*/}
outfn="${outfn%.*}_ll.h"

outfile=/tmp/$outfn
if [ -f $outfile ] ; then 
  echo removing old version of $outfile
  rm -f $outfile
fi
touch $outfile

echo "//" >> $outfile
echo "// This file $outfn was created by $0 " >> $outfile
echo "// from input file: $llfile  " >> $outfile
echo "// on `date` " >> $outfile
echo "//" >> $outfile
echo "#ifndef INLINE " >> $outfile
echo "#define INLINE " >> $outfile
echo "#endif" >> $outfile
echo "#ifndef _LL_GLOBAL_" >> $outfile
echo "#define _LL_GLOBAL_ " >> $outfile
echo "#endif" >> $outfile
echo "#ifndef _LL_SHARED_" >> $outfile
echo "#define _LL_SHARED_ " >> $outfile
echo "#endif" >> $outfile

# Edits:
#  i32 -> unsigned int 
#  i64 -> unsigned long long int
#  @ f -> ""
#  addrspace(1) -> _LL_GLOBAL_
#  addrspace(3) -> _LL_SHARED_
#  define -> INLINE
#  { -> ;
#  #[0-9] -> ""
#  % -> ""
#  i8 -> unsigned char
#  i16  -> unsigned short int
#  i1 -> bool 
#  local_unnamed_addr -> ""
#  FIXME: Add const where necessary
#  FIXME: convert nocapture to ???
#  FIXME: if define is not all on a single line 
#  FIXME: Create data structure externs from globals.  Need to look at more than ll defines. 
#  FIXME: convert readonly to ???  const ? 

grep define $llfile | grep "{" | sed "s/define/INLINE/" | sed "s/i32/unsigned int/g" | sed "s/i64/unsigned long long int/g" | sed "s/%//g"  | sed "s/{/;/g" | sed "s/#[0-9]//g" | sed "s/addrspace(3)/_LL_SHARED_ /g" | sed "s/addrspace(1)/_LL_GLOBAL_ /g"  | sed "s/@//g" | sed "s/i8/unsigned char/g" |  sed "s/i16/unsigned short int/g" | sed "s/i1/bool/g" | sed "s/local_unnamed_addr//g" | tee -a $outfile

echo 
echo outfile is $outfile
echo 
