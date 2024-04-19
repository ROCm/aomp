#!/bin/bash

################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

# Save current directory because this script does a cd
curdir=$PWD

# Get the directory where the example scripts and source are located
# This can be read-only.  Nothing is wrtten to the examples directory
_zero_dir=`dirname $0`
cd $_zero_dir
RO_SHELL_DIR=$PWD

# This is the test run directory in /tmp
TEST_DIR=/tmp/$USER/roctrace_test

BIN_DIR=$TEST_DIR/bin
mkdir -p $BIN_DIR
echo
echo "+===================================================+"
echo "|    Building test binaries in directory:           |"
echo "|    $BIN_DIR "
echo "+===================================================+"
echo

cp -rp $RO_SHELL_DIR/MatrixTranspose $TEST_DIR/.
cp -rp $RO_SHELL_DIR/MatrixTranspose_test $TEST_DIR/.
cp -rp $RO_SHELL_DIR/../../Makefile.find_gpu_and_install_dir $TEST_DIR/.

[ -z $AOMP ] && AOMP=/opt/rocm/lib/llvm
[ -z $LLVM_INSTALL_DIR ] && LLVM_INSTALL_DIR=$AOMP
#  For installed rocm llvm, avoid issues with symbolic link to old llvm install dir
[ "$LLVM_INSTALL_DIR" == "/opt/rocm/llvm" ] && [ -L $LLVM_INSTALL_DIR ] && [ -d /opt/rocm/lib/llvm ] && export LLVM_INSTALL_DIR=/opt/rocm/lib/llvm

AOMP=$LLVM_INSTALL_DIR

if [ -d $LLVM_INSTALL_DIR/../roctracer ] ; then
  # LLVM_INSTALL_DIR is a rocm compiler install now in /opt/rocm/lib/llvm as of ROCM 6.1
  INC_PATH=$LLVM_INSTALL_DIR/../../include/roctracer
  LIB_PATH=$LLVM_INSTALL_DIR/..
  ROCM_PATH=$LLVM_INSTALL_DIR/../..
  ALL_LIB_PATH=$ROCM_PATH/lib:$LIB_PATH:$LLVM_INSTALL_DIR/../roctracer
  DEVICE_LIB_PATH=$LLVM_INSTALL_DIR/../../amdgcn/bitcode
else
  # LLVM_INSTALL_DIR is build of standalone install of AOMP
  INC_PATH=$LLVM_INSTALL_DIR/include/roctracer
  LIB_PATH=$LLVM_INSTALL_DIR/lib
  ROCM_PATH=$LLVM_INSTALL_DIR
  ALL_LIB_PATH=$LIB_PATH:$LLVM_INSTALL_DIR/lib/roctracer
  DEVICE_LIB_PATH=$LLVM_INSTALL_DIR/amdgcn/bitcode
fi

EXPORT_ENV="HIP_VDI=1 ROCM_PATH=${ROCM_PATH} HSA_PATH=${ROCM_PATH} INC_PATH=${INC_PATH} LIB_PATH=${LIB_PATH} HIP_CLANG_PATH=${LLVM_INSTALL_DIR}/bin DEVICE_LIB_PATH=${DEVICE_LIB_PATH} HIPCC_VERBOSE=3"

echo export $EXPORT_ENV
export $EXPORT_ENV

# BIN 1 
make -C $TEST_DIR/MatrixTranspose
cp ${TEST_DIR}/MatrixTranspose/MatrixTranspose ${BIN_DIR}/MatrixTranspose

# BIN 2  test
make -C "${TEST_DIR}/MatrixTranspose_test"
cp ${TEST_DIR}/MatrixTranspose_test/MatrixTranspose ${BIN_DIR}/MatrixTranspose_test

# BIN 3  test
HIP_API_ACTIVITY_ON=1 make -C "${TEST_DIR}/MatrixTranspose_test"
cp ${TEST_DIR}/MatrixTranspose_test/MatrixTranspose ${BIN_DIR}/MatrixTranspose_hipaact_test

# BIN 4  test 
MGPU_TEST=1 make -C "${TEST_DIR}/MatrixTranspose_test"
cp ${TEST_DIR}/MatrixTranspose_test/MatrixTranspose ${BIN_DIR}/MatrixTranspose_mgpu

# BIN 5  test 
C_TEST=1 make -C "${TEST_DIR}/MatrixTranspose_test"
cp ${TEST_DIR}/MatrixTranspose_test/MatrixTranspose ${BIN_DIR}/MatrixTranspose_ctest

cp ${RO_SHELL_DIR}/golden_traces/*_trace.txt ${BIN_DIR}/
cp ${RO_SHELL_DIR}/golden_traces/tests_trace_cmp_levels.txt ${BIN_DIR}/

# for now we do not check the traces and do not run hsa tests.
check_trace_flag=0
run_hsa_tests=0
echo
echo "+===================================================+"
echo "|  Done building test binaries                      |"
echo "|  (some binaries are reused)                       |"
echo "|  Now Running tests with roctracer                 |"
echo "|  check_trace_flag: $check_trace_flag                              |"
echo "|  run_hsa_testsd:   $run_hsa_tests                              |"
echo "|    cd $TEST_DIR  "
echo "+===================================================+"
echo
cd $TEST_DIR

# enable tools load failure reporting
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
# paths to ROC profiler and other libraries
export LD_LIBRARY_PATH=$ALL_LIB_PATH

# test filter input
test_filter=-1

if [ -n "$1" ] ; then
  test_filter=$1
  shift
fi
if [ "$2" = "-n" ] ; then
  check_trace_flag=0
fi

test_status=0
test_runnum=0
test_number=0
failed_tests="Failed tests:"

xeval_test() {
  test_number=$test_number
}

eval_test() {
  label=$1
  cmdline=$2
  test_name=$3
  test_trace=$test_name.txt

  if [ $test_filter = -1  -o $test_filter = $test_number ] ; then
    echo "test $test_number: $test_name \"$label\""
    echo "CMD: \"$cmdline\""
    test_runnum=$((test_runnum + 1))
    eval "$cmdline" >$test_trace 2>&1
    is_failed=$?
    if [ $is_failed != 0 ] ; then
      cat $test_trace
      echo "$test_name: FAILED"
    else 
      python3 $RO_SHELL_DIR/check_trace.py -in $test_name -ck $check_trace_flag -rd $TEST_DIR
      is_failed=$?
      if [ $is_failed != 0 ] ; then
        echo "Trace checker error:"
        python3 $RO_SHELL_DIR/check_trace.py -v -in $test_name -ck $check_trace_flag -rd $TEST_DIR
        echo "$test_name: FAILED"
      else 
        echo "$test_name: PASSED"
      fi
    fi
    if [ $is_failed != 0 ] ; then
      failed_tests="$failed_tests\n  $test_number: $test_name - \"$label\""
      test_status=$(($test_status + 1))
    fi
  fi

  test_number=$((test_number + 1))
}

# Standalone test
# rocTrecer is used explicitely by test
eval_test "standalone C test" "./bin/MatrixTranspose_ctest" MatrixTranspose_ctest_trace
eval_test "standalone HIP test" "./bin/MatrixTranspose_test" MatrixTranspose_test_trace
eval_test "standalone HIP hipaact test" "./bin/MatrixTranspose_hipaact_test" MatrixTranspose_hipaact_test_trace
eval_test "standalone HIP MGPU test" "./bin/MatrixTranspose_mgpu" MatrixTranspose_mgpu_trace

# Tool test
# rocTracer/tool is loaded by HSA runtime
if [ -d $LLVM_INSTALL_DIR/../roctracer ] ; then
  export HSA_TOOLS_LIB="$LLVM_INSTALL_DIR/../lib/roctracer/libtracer_tool.so"
else
  export HSA_TOOLS_LIB="$LLVM_INSTALL_DIR/lib/roctracer/libtracer_tool.so"
fi

# SYS test
export ROCTRACER_DOMAIN="sys:roctx"
eval_test "tool SYS test" ./bin/MatrixTranspose MatrixTranspose_sys_trace
export ROCTRACER_DOMAIN="sys:hsa:roctx"
eval_test "tool SYS/HSA test" ./bin/MatrixTranspose MatrixTranspose_sys_hsa_trace
# Tracing control <delay:length:rate>
export ROCTRACER_DOMAIN="hip"
eval_test "tool period test" "ROCP_CTRL_RATE=10:100000:1000000 ./bin/MatrixTranspose" MatrixTranspose_hip_period_trace
eval_test "tool flushing test" "ROCP_FLUSH_RATE=100000 ./bin/MatrixTranspose" MatrixTranspose_hip_flush_trace

if [[ "$run_hsa_tests" == 1 ]] ; then 
  # HSA test
  export ROCTRACER_DOMAIN="hsa"
  # test trace
  export ROC_TEST_TRACE=1
  # kernels loading iterations
  export ROCP_KITER=1
  # kernels dispatching iterations per kernel load
  # dispatching to the same queue
  export ROCP_DITER=1
  # GPU agents number
  export ROCP_AGENTS=1
  # host threads number
  # each thread creates a queue pre GPU agent
  export ROCP_THRS=1

  eval_test "tool HSA test" ./bin/hsa/ctrl ctrl_hsa_trace

  echo "<trace name=\"HSA\"><parameters api=\"hsa_agent_get_info, hsa_amd_memory_pool_allocate\"></parameters></trace>" > input.xml
  export ROCP_INPUT=input.xml
  eval_test "tool HSA test input" ./bin/hsa/ctrl ctrl_hsa_input_trace
fi

#valgrind --leak-check=full $tbin
#valgrind --tool=massif $tbin
#ms_print massif.out.<N>

echo
echo "+===================================================+"
echo "|  $test_number tests total / $test_runnum tests run / $test_status tests failed     |"
if [ $test_status != 0 ] ; then
  echo "|    $failed_tests "
fi
echo "|  check_trace_flag: $check_trace_flag                              |"
echo "|  run_hsa_testsd:   $run_hsa_tests                              |"
echo "| See traces in directory: $BIN_DIR "
echo "+===================================================+"
echo

cd $curdir
exit $test_status
