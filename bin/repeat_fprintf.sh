#!/bin/bash
#
# repeat_fprintf.sh <gfxid> <iters> <compiler_bin_dir>
#    Compile and repeat execute the embedded source in this file. 
#
# Default args: gfx90a 100 /opt/rocm/llvm/bin
_gfxid=${1:-gfx90a}
_iters=${2:-100}
_compiler_bin_dir=${3:-/opt/rocm/llvm/bin}

# Check default gfxid using offload-arch
if [ -f $_compiler_bin_dir/offload-arch ] ; then 
  _oa_gfxid=`$_compiler_bin_dir/offload-arch`
  if [ "$_oa_gfxid" != "$_gfxid" ] ; then 
    echo "WARNING: changing gfxid to $_oa_gfxid"
    _gfxid=$_oa_gfxid
  fi
fi

# Cleanup old source and binary files
_source_file="/tmp/fprintf.c"
_binary="/tmp/fprintf"
[ -f $_binary ] && rm $_binary
[ -f $_source_file ] && rm $_source_file
# Recreate source file from this embedded source
/bin/cat >$_source_file  <<"EOF"
#include <stdio.h>
#include <omp.h>

void write_index(int*a, int N, FILE* fileptr ){
   #pragma omp target teams distribute parallel for map(tofrom: a[0:N]) is_device_ptr(fileptr)
   for(int i=0;i<N;i++) {
      fprintf(fileptr, "fprintf: updating a[%d] addr:%p  file ptr:%p\n",i,&a[i], fileptr);
      a[i]=i;
   }
}

int main(){
    const int N = 10;    
    int a[N],validate[N];
    for(int i=0;i<N;i++) {
        a[i]=0;
        validate[i]=i;
    }
    #pragma omp target map(tofrom: a[0:N]) 
    printf("Warmup kernel\n");

    FILE* fileptr = fopen("/tmp/gpu.log", "w");
    write_index(a,N,fileptr);
    fclose(fileptr);

    int flag = -1;
    for(int i=0;i<N;i++) {
      if(a[i]!=validate[i]) {
//      print 1st bad index
        if( flag == -1 ) 
          printf("First fail: a[%d](%d) != validate[%d](%d)\n",
			  i,a[i],i,validate[i]);
        flag = i;
      }
    }

    if( flag == -1 ){
        printf("Success\n");
        return 0;
    } else {
        printf("Last fail: a[%d](%d) != validate[%d](%d)\n",
			flag,a[flag],flag,validate[flag]);
        return 1;
    }
}
EOF

function run_tests() {
   _log=stdout.log
   _rc0=0
   for i in `seq 1 $_iters` ; do $_binary ; [ $? == 0 ] && _rc0=$(( $_rc0 + 1 )) ; done >$_log
   _fails=$(( $_iters - $_rc0 ))
   _failrate=$(( ( $_fails * 100 ) / $_iters ))
}

_compile_cmd="$_compiler_bin_dir/clang -O2 -fopenmp --offload-arch=$_gfxid $_source_file -o $_binary"
echo
echo $_compile_cmd
$_compile_cmd
[ ! -f $_binary ] && echo "compile fail" && exit 1 

echo
echo "Testing $_binary for $_iters iterations ..."
run_tests
echo "Iterations:$_iters  Success:$_rc0  Fails:$_fails  Failrate:${_failrate}%"

echo
echo "Testing $_binary for $_iters iterations with HSA_ENABLE_SDMA=0..."
export HSA_ENABLE_SDMA=0
run_tests
echo "Iterations:$_iters  Success:$_rc0  Fails:$_fails  Failrate:${_failrate}%"
