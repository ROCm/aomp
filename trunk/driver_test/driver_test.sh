#!/bin/bash
#
# driver_test.sh : create two heterogeneous objects and link to main c code. 
#                  to demonstrate offloading driver. 
#
# Three environment variables are used by this script:
#    TRUNK:  Location of LLVM trunk installation, default is $HOME/rocm/trunk
#    OARCH:  Offload architecture, default is sm_70 
#    USE_FLANG: Use flang-new, default is NO because flang offload driver still broken
#
# Example: To test flang-new with AMD gpu gfx908 run this command:
#   OARCH=gfx908 USE_FLANG=YES ./driver_test.sh  

# Set environment variable defaults here:
TRUNK=${TRUNK:-$HOME/rocm/trunk}
OARCH=${OARCH:-sm_70}
USE_FLANG=${USE_FLANG:-NO}

_llvm_bin_dir=$TRUNK/bin

#extra_args="-v -save-temps"
extra_args=""

# Generate driver cmds for each of three steps
$_llvm_bin_dir/flang-new -### $extra_args -c -fopenmp --offload-arch=$OARCH dec_arrayval.f95 -o dec_arrayval.o 2>flang-new.cmds
$_llvm_bin_dir/clang -### $extra_args -c -fopenmp --offload-arch=$OARCH dec_arrayval.c -o dec_arrayval.o 2>clang.cmds
$_llvm_bin_dir/clang -### $extra_args -fopenmp --offload-arch=$OARCH inc_arrayval.o dec_arrayval.o main.c -o main 2>main.cmds

# The increment function shows steps to build heterogeneous object with c
cmd1="$_llvm_bin_dir/clang $extra_args -c -fopenmp --offload-arch=$OARCH inc_arrayval.c -o inc_arrayval.o"
echo $cmd1
$cmd1

[ -f dec_arrayval.o ] && rm dec_arrayval.o
if [ "$USE_FLANG" == "NO" ] ; then
   echo "WARNING: NOT compiling dec_arrayval.f95, Compiling dec_arrayval.c insted to test execution"
   echo "         Set USE_FLANG=YES to compile/test flang-new on  dec_arrayval.f95"
   cmd2="$_llvm_bin_dir/clang $extra_args -c -fopenmp --offload-arch=$OARCH dec_arrayval.c -o dec_arrayval.o"
else
   cmd2="$_llvm_bin_dir/flang-new $extra_args -c -fopenmp --offload-arch=$OARCH dec_arrayval.f95 -o dec_arrayval.o"
fi
echo $cmd2
$cmd2
if [ $? == 0 ] ; then 
   [ -f main ] && rm main
   compile_main_cmd="$_llvm_bin_dir/clang -fopenmp --offload-arch=$OARCH inc_arrayval.o dec_arrayval.o main.c -o main"
   echo $compile_main_cmd
   $compile_main_cmd
   echo ./main
   ./main
   echo OMP_TARGET_OFFLOAD=MANDATORY ./main
   export OMP_TARGET_OFFLOAD=MANDATORY
   OMP_TARGET_OFFLOAD=MANDATORY ./main
fi

