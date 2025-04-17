#!/bin/bash
# Copyright(C) 2020 Advanced Micro Devices, Inc. All rights reserved.
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
#  kokkos_build.sh: Script to run Kokkos unit tests and get statistics
#                   This will run kokkos in directory $HOME/kokkos_build.<GPUNAME>
#
#  Written by Jan-Patrick Lehr JanPatrick.Lehr@amd.com
#
PROGVERSION="X.Y-Z"
#

function print_info() {
  toPrint="$1"

  echo "[Info]: $toPrint"
}
function print_error() {
  toPrint="$1"

  echo "[Error]: $toPrint"
}

function charexists() {
  char="$1"; shift
  case "$*" in *"$char"*) return;; esac; return 1
}

SCRIPT_REALPATH=`realpath $0`
TOOLDIR=`dirname $SCRIPT_REALPATH`

## To get consistent GPU names for building and running we rely on AOMP here as well
echo "$AOMP"
AOMP="${AOMP:-_AOMP_INSTALL_DIR_}"
echo "$AOMP"
if [ ! -d $AOMP ] ; then
   print_error "AOMP is not installed in ${AOMP}. Please set the environment variable."
   exit 1
fi

if [[ "$AOMP" =~ "opt" ]]; then
  # xargs trims the string off whitespaces
  export DETECTED_GPU=$($AOMP/../bin/rocminfo | grep -m 1 -E gfx[^0]{1}.{2} | awk '{print $2}')
  print_info "Set AOMP_GPU with rocminfo: $DETECTED_GPU"
else
  if [ -a $AOMP/bin/rocminfo ]; then
    print_info "Set AOMP_GPU with rocminfo"
    export DETECTED_GPU=$($AOMP/bin/rocminfo | grep -m 1 -E gfx[^0]{1}.{2} | awk '{print $2}')
  else
    print_info "Set AOMP_GPU with offload-arch."
    export DETECTED_GPU=$($AOMP/bin/offload-arch | grep -m 1 -E gfx[^0]{1}.{2})
  fi
fi

COMPILERNAME_TO_USE=${_COMPILER_TO_USE_:-clang++}
AOMP_VERSION=$($AOMP/bin/${COMPILERNAME_TO_USE} --version | head -n 1)

AOMP_GPU=${AOMP_GPU:-$DETECTED_GPU}

GIT_DIR=${GIT_DIR:-$HOME/git}
KOKKOS_BUILD_PREFIX=${KOKKOS_BUILD_PREFIX:-$HOME}
KOKKOS_TAG=${_KOKKOS_TAG_:-'NA'}

if [ "$1" == "hip" ] ; then
   kokkos_backend="hip"
   KOKKOS_BUILD_DIR=${KOKKOS_BUILD_DIR:-$KOKKOS_BUILD_PREFIX/kokkos-${KOKKOS_TAG}_build_hip.$AOMP_GPU}
   KOKKOS_INSTALL_DIR=${KOKKOS_INSTALL_DIR:-$KOKKOS_BUILD_PREFIX/kokkos-${KOKKOS_TAG}_hip.$AOMP_GPU}
else
   kokkos_backend="openmp"
   KOKKOS_BUILD_DIR=${KOKKOS_BUILD_DIR:-$KOKKOS_BUILD_PREFIX/kokkos-${KOKKOS_TAG}_build_omp.$AOMP_GPU}
   KOKKOS_INSTALL_DIR=${KOKKOS_INSTALL_DIR:-$KOKKOS_BUILD_PREFIX/kokkos-${KOKKOS_TAG}_omp.$AOMP_GPU}
fi

KOKKOS_EXAMPLES_SOURCE_DIR=${KOKKOS_EXAMPLES_SOURCE_DIR:-$GIT_DIR/kokkos-openmptarget-examples}

## We want to process the command line arguments
# The script accepts the arguments 'unittest' and 'cgsolve' for now
# It sets corresponding variables to 'yes' that indicate if these tests should
# be executed or not
KOKKOS_RUN_UNIT_TEST='no'
KOKKOS_RUN_CGSOVLE='no'

# We support summary and detail execution
KOKKOS_RUN_TYPE=${KOKKOS_RUN_TYPE:-summary}

# For the CI runs we want to move the summary to the CI folder
KOKKOS_EXTRACT_FILE_LOCATION=${KOKKOS_EXTRACT_DIRECTORY:-$AOMP_OPENMP_CI}

if [ "$#" -eq 0 ]; then
  print_error "Please indicate what to run 'unittest', 'cgsolve'"
  exit 1
fi

while (( "$#" )); do
  if [ "$1" == 'unittest' ]; then
    KOKKOS_RUN_UNIT_TEST='yes'
  elif [ "$1" == 'cgsolve' ]; then
    KOKKOS_RUN_CGSOVLE='yes'
  fi
  shift
done

print_info "Selecting what to run unit-test ($KOKKOS_RUN_UNIT_TEST) or cgsolve ($KOKKOS_RUN_CGSOVLE)"


if [ $KOKKOS_RUN_UNIT_TEST == 'yes' ]; then
  if [ -z $KOKKOS_EXTRACT_FILE_LOCATION ]; then
    initialDir=$PWD
  else
    initialDir=$KOKKOS_EXTRACT_FILE_LOCATION
  fi

  cd $KOKKOS_BUILD_DIR || exit 1
  
  cd core/unit_test || exit 1
  
  # Run the top-level summary version of the tests
  if [ "$KOKKOS_RUN_TYPE" == "summary" ]; then
    OMP_NUM_THREADS=2 ctest --timeout 240 -j 4
    print_info "For more details, please set KOKKOS_RUN_TYPE to 'detail'"
 
  elif [ "$KOKKOS_RUN_TYPE" == "detail" ]; then
    # For a detail run, we follow the procedure
    # 1. Search for all executables in the unit_test directory.
    # 2. Query each executable for its testsuites and the contained test cases.
    # 3. Run an executable separately for each testsuite.testcase to detect and cover individual hangs (and still generate results).
    #    Put each individual result into its own file to merge them back afterwards. 

    # Start with capturing a list of all executables.
    declare -a EXE_FILES
    for EXE in $(find . -maxdepth 1 -perm -111 -type f); do
      echo "AOMPCI LOG: $PWD / $EXE"
      EXE_FILES+=("$EXE")
    done

    # Rmove previous test results
    find . -iname "RESULT_*" -delete

    # 2. Query each executable for all contained test cases
    for UT in "${EXE_FILES[@]}"; do
      echo "Executable $UT"
      # Have gtest list all available test suites and the test cases therein.
      # A test suite name ends with a '.', so filter for that to differentiate between them.
      # Run each test case individually under time limit to detect potential hang and report as test failure, while continue to run
      # all the other test cases normally.
      for TC in $(python3 $TOOLDIR/extractTestsFromGTest.py $UT); do
        echo "Running ${UT} TC: $TC"
        fName=${UT/KokkosCore/RESULT}_${TC/\//_}
        OMP_PROC_BIND=spread OMP_NUM_THREADS=2 timeout 2m ${UT} --gtest_filter="$TC" --gtest_output=json:$PWD/${fName}.json
        # Check for return code of command, 124 indicating timeout (interpreted as hang)
        if [ "$?" == "124" ]; then
          echo "Command timed out. Need to handle"
        fi
      done
    done

    # The AOMP_CI_ACCUMULATOR is a Python script to read the resulting GTest json files and transform them into extract-format files
    if [ ! -z "$AOMP_CI_ACCUMULATOR" ]; then
      tmpResFile=accuResult
      # Collect everything in the KOKKOS_INTERMEDIATE_FILE ${tmpResFile} is not touched during this command
      # I know that this looks a bit clumsy and it can sure be improved later.
      find . -iname "RESULT_*" -exec python3 ${AOMP_CI_ACCUMULATOR} --snapshot ${KOKKOS_FAILS_SNAPSHOT} --failfile ${KOKKOS_FAIL_FILE} --intermediate-file ${KOKKOS_INTERMEDIATE_FILE} {} ${tmpResFile} \;
      # Generate the extract from the KOKKOS_INTERMEDIATE_FILE
      python3 ${AOMP_CI_ACCUMULATOR} --snapshot ${KOKKOS_FAILS_SNAPSHOT} --failfile ${KOKKOS_FAIL_FILE} ${KOKKOS_INTERMEDIATE_FILE} ${tmpResFile}
      cp ${tmpResFile}-perf.ext ${initialDir}/accumulatedResults-pass-rate-${KOKKOS_TAG}.ext
      cp ${tmpResFile}-corr.ext ${initialDir}/accumulatedResults-corr-${KOKKOS_TAG}.ext
      rm ${tmpResFile}-corr.ext ${tmpResFile}-perf.ext ${KOKKOS_INTERMEDIATE_FILE}
    fi

  else
    print_error "Please set KOKKOS_RUN_TYPE to summary or detail"
  fi
fi

if [ $KOKKOS_RUN_CGSOVLE == 'yes' ]; then
  print_info "Running performance cgsolve"
  # Switch to the example directory
  cd $KOKKOS_EXAMPLES_SOURCE_DIR/cgsolve || exit 1

  # The example runs both an OpenMP target version of cgsolve and a version using the Kokkos library
  # with the OpenMP target backend
  # We execute an increased problem size and require higher precision to provoke longer runtimes.
  print_info "Running default (small) problem"
  ./cgsolve.ompt

  echo ""

  print_info "Running increased (large) problem"
  ./cgsolve.ompt 400 300 0.000000001 
fi
