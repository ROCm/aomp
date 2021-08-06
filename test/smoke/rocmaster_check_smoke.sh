#!/bin/bash
#
# Checks all tests in smoke directory using make check. Programs return 0 for success or a number > 0 for failure.
# Tests that need to be visually inspected: devices, pfspecify, pfspecify_str, stream
#
#

echo "This script is deprecated. Please run : cd aomp/bin; ./run_rocm_test.sh"

if [ "$EPSDB_ROCMASTER_OVERRIDE" != "1" ]; then
  exit 1
fi

#default to on for qa runs
EPSDB=${EPSDB:-1}

if [ "$EPSDB" == "1" ]; then
  export AOMP=/opt/rocm/llvm
  export AOMP_GPU=`$AOMP/../bin/mygpu`
else
  export AOMP=/opt/rocm/aomp
fi

cleanup(){
  rm -rf check-smoke.txt
  rm -rf passing-tests.txt
  rm -rf failing-tests.txt
  rm -rf make-fail.txt
}

#Clean all testing directories
make clean
cleanup

path=$(pwd)
echo ""
echo "RUNNING ALL TESTS IN: $path"
echo ""

echo "************************************************************************************" > check-smoke.txt
echo "                   A non-zero exit code means a failure occured." >> check-smoke.txt
echo "***********************************************************************************" >> check-smoke.txt

skiptests="devices pfspecifier pfspecifier_str target_teams_reduction hip_rocblas tasks reduction_array_section targ_static omp_wtime data_share2 global_allocate complex2 flang_omp_map omp_get_initial slices printf_parallel_for_target reduction_shared_array d2h_slow_copy"

if [ "$EPSDB" == "1" ]; then
  skiptests+=" taskwait_prob flang_isystem_prob flang_real16_prob"
  # additional for rocm 4.4
  skiptests+=" flang-272730-complex flang-273990-2 flang-tracekernel math_exp math_libmextras simple_ctor use_device_ptr"
  # additional for rocm 4.3
  skiptests+=" alignedattribute devito_prob1 libgomp-292348 math_max"

fi

# amd-stg-open only has commits up to 08/11/20, which does not include these fixes for gfx908
if [ "$EPSDB" == "1" ] && [ "$AOMP_GPU" == "gfx908" ];then
  skiptests+=" red_bug_51 test_offload_macros"
fi

#Loop over all directories and make run / make check depending on directory name
for directory in ./*/; do
  (cd "$directory" && path=$(pwd) && base=$(basename $path)
     skip=0
     #if [[ $skiptests == " $base " ]] ; then
     for test in $skiptests ; do
       if [ $test == $base ] ; then 
         skip=1
         break
       fi
     done
    if [ $skip -ne 0 ] ; then
      echo "Skip $base!"
    
    #flags has multiple runs
    elif [ $base == 'flags' ] ; then
      make
      make run > /dev/null 2>&1
    else
      make
      if [ $? -ne 0 ]; then
        echo "$base: Make Failed" >> ../make-fail.txt
      fi
      make check > /dev/null 2>&1
      #liba_bundled has an additional Makefile, that may fail on the make check
      if [ $? -ne 0 ] && ( [ $base == 'liba_bundled' ] || [ $base == 'liba_bundled_cmdline' ] ) ; then
        echo "$base: Make Failed" >> ../make-fail.txt
      fi
    fi
    echo ""
    )
	
done

echo ""
if [ -e check-smoke.txt ]; then
  cat check-smoke.txt
fi
if [ -e make-fail.txt ]; then
  cat make-fail.txt
fi
echo ""

#Gather Test Data
if [ -e passing-tests.txt ]; then
  ((total_tests=$(wc -l <  passing-tests.txt)))
fi
if [ -e make-fail.txt ]; then
  ((total_tests+=$(wc -l <  make-fail.txt)))
fi
if [ -e failing-tests.txt ]; then
  ((total_tests+=$(wc -l <  failing-tests.txt)))
fi

#Print Results
echo -------------------- Results --------------------
echo "Number of tests: $total_tests"
echo ""
echo Passing tests: `wc -l <  passing-tests.txt`/$total_tests
echo ""

#Print failed tests
if [ -e failing-tests.txt ]; then
  echo "Runtime Fails"
  echo "--------------------"
  cat failing-tests.txt
  echo ""
fi

if [ -e make-fail.txt ]; then
  echo "Compile Fails"
  echo "--------------------"
  cat make-fail.txt
fi
