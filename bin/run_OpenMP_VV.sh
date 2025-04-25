#!/bin/bash
#
#  run_openmpvv.sh:
#    Script for running the OpenMP_VV suite found at
#    https://github.com/OpenMP-Validation-and-Verification/OpenMP_VV
#    We expect the repository to be located within 'AOMP_REPOS_TEST'
#
#    By default, all test cases will be compiled & executed. Then the results
#    will be processed into reports for further evaluation. Some tests might
#    get excluded automatically, depending on the available hardware.
#

ulimit -t 120

AOMP_OPENMPVV_REPO_NAME=${AOMP_OPENMPVV_REPO_NAME:-OpenMP_VV}

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "${realpath}")
export AOMP_USE_CCACHE=0

. ${thisdir}/aomp_common_vars
# --- end standard header ----

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}
FLANG=${FLANG:-flang}

# Use function to set and test AOMP_GPU
setaompgpu

# OpenMP_VV might discard absolute paths to compiler binaries, hence add the containing directory to PATH.
AOMP_BIN=${AOMP_BIN:-"${AOMP}/bin"}
export PATH="${AOMP_BIN}:${PATH}"
export CXX_VERSION=""
export C_VERSION=""
export F_VERSION=""

# OpenMP_VV specific variables
openmpvv_dir=${AOMP_REPOS_TEST}/${AOMP_OPENMPVV_REPO_NAME}
openmpvv_compile_flags=${openmpvv_compile_flags:-"-O2 -fopenmp --offload-arch=${AOMP_GPU}"}
openmpvv_logging_flags="LOG=1 LOG_ALL=1 VERBOSE_TESTS=1 VERBOSE=1"

# Trigger creation of reports
function make_OpenMP_VV_reports() {
  # Lines for report_summary tail
  numlines=4
  if [ "$1" == "52" ]; then
    cpp_files=$(find "${openmpvv_dir}/tests/5.2" -type f -name '*cpp')
    if [ "${cpp_files}" == "" ]; then
      numlines=3
    fi
  fi

  # Start reports
  make report_html
  make report_summary >> combined-results.txt
  make report_summary  | tail -"${numlines}" >> abrev.combined-results.txt
  mv results_report results_report"$1"
}

# Prepare and execute the make command.
# Using provided (1) OpenMP version and (2) optional sources, gather all needed
# parameters and combine them into an executable make command. Then perform make
# tidy, followed by the actual make.
function perform_make() {
  # Capture OpenMP version and custom sources.
  local omp_version="$1"
  # We will need to account for empty values upon command creation.
  local sources="$2"

  # If no OpenMP version was provided: abort script.
  if [[ -z "${omp_version}" ]]; then
    echo "error: '${FUNCNAME[0]}' invoked without an OMP version"
    exit 1
  fi

  # Update make parameters based on current values.
  local cc_path="CC=${AOMP_BIN}/clang"
  local cxx_path="CXX=${AOMP_BIN}/clang++"
  local fc_path="FC=${AOMP_BIN}/${FLANG}"
  local cflags="CFLAGS=-lm ${openmpvv_compile_flags}"
  local cxxflags="CXXFLAGS=${openmpvv_compile_flags}"
  local fflags="FFLAGS=${openmpvv_compile_flags}"
  local logging_flags

  # Prepare each logging flag as an array element.
  read -r -a logging_flags <<< "${openmpvv_logging_flags}"

  # Prepare make command.
  make_command=(
    make
    "${cc_path}"
    "${cxx_path}"
    "${fc_path}"
    "${cflags}"
    "${cxxflags}"
    "${fflags}"
    "${logging_flags[@]}"
    "${omp_version}"
  )

  # Conditionally add sources.
  if [[ -n "${sources}" ]]; then
    make_command+=("${sources}")
  fi

  # Append final 'all' target (= Build and run sources).
  make_command+=("all")

  # Print the command.
  echo "Executing command:"
  echo "${make_command[*]}"

  # Tidy up, then execute the make command.
  make tidy
  "${make_command[@]}"
}

# Skip unified_shared_memory and unified_address tests as they render gfx 906/900 unusable.
if [ "${SKIP_USM}" == "1" ]; then
   openmpvv_test_filter="\" -type f ! \( -name *unified_shared_memory* -o -name *unified_address* \)\""
fi

# Get the single testcase to execute, if requested by the user.
testcase="$1"
if [ "${testcase}" ]; then
  # Search for matching filenames and store results as array.
  # Note: find command is executed in subshell and redirected as file.
  mapfile -t matches < <(find "${openmpvv_dir}/tests" -wholename "*${testcase}*" -type f)
  count="${#matches[@]}"

  # Check if file exists and is unique, if not: exit.
  if [ ${count} != 1 ]; then
    echo "ERROR: Trying to run a single OPENMP_VV test case: '${testcase}'"
    echo "       A single unique file could not be found in ${openmpvv_dir}/tests"
    echo "       Found ${count} matches with the same filename"
    # If the requested file is not unique, show the matches.
    # Additionally, print a potentially working example.
    if [ ${count} -gt 1 ]; then
      for file in "${matches[@]}"; do
        echo "         ${file}"
      done
      # Build suggestion: Remove prefix "${openmpvv_dir}/tests/"
      example=${matches[0]//${openmpvv_dir}\/tests\//}
      # Multiple matches: Provide specific suggestion
      echo "       For example, try this command:"
      echo "         $0 ${example}"
    else
      # Zero matches: Provide generic suggestion
      echo "       For example, try this command:"
      echo "         $0 test_target_teams_distribute_parallel_for.c"
    fi
    exit 1
  fi

  # This will get a single valid filename for SOURCES= arg on make command below.
  testcase_file_path=${matches[0]}
  reldir=${matches[0]#${openmpvv_dir}/tests/}
  testcase_omp_version=${reldir%%/*}
else
  make_target="all"
fi

if [ "${ROCMASTER}" == "1" ]; then
  ./clone_test.sh
  pushd "${openmpvv_dir}" || exit
    # Lock at specific hash for consistency
    git reset --hard 0fbdbb9f7d3b708eb0b5458884cfbab25103d387
  popd || exit
else
  pushd "${openmpvv_dir}" || exit
  git pull
  popd || exit
fi

pushd "${openmpvv_dir}" || exit

# Clean up previously generated reports and binaries.
if [ "${make_target}" == "all" ] || [ "${openmpvv_test_filter}" != "" ]; then
   [ -d results_report45 ] && rm -rf results_report45
   [ -d results_report50 ] && rm -rf results_report50
   [ -d results_report51 ] && rm -rf results_report51
   [ -d results_report52 ] && rm -rf results_report52
   [ -f combined-results.txt ] && rm -f combined-results.txt
   [ -f abrev.combined-results.txt ] && rm -f abrev.combined-results.txt
   make tidy
fi

# Run user-requested testcase.
if [ "${make_target}" != "all" ]; then
  echo
  pwd
  echo
  echo "START: Single OPENMP_VV test case: ${testcase}"
  echo "       Source file:  ${testcase_file_path}"
  echo "       OMP_VERSION:  ${testcase_omp_version}"
  if [ -f "${openmpvv_dir}/bin/${testcase}.o" ]; then
     echo "       rm ${openmpvv_dir}/bin/${testcase}.o"
     rm "${openmpvv_dir}/bin/${testcase}.o"
  fi
  if [ "${testcase_omp_version}" == "5.0" ]; then
     export openmpvv_compile_flags="${openmpvv_compile_flags} -fopenmp-version=50"
  elif [ "${testcase_omp_version}" == "5.1" ]; then
     export openmpvv_compile_flags="${openmpvv_compile_flags} -fopenmp-version=51"
  elif [ "${testcase_omp_version}" == "5.2" ]; then
     export openmpvv_compile_flags="${openmpvv_compile_flags} -fopenmp-version=52"
  elif [ "${testcase_omp_version}" == "4.5" ]; then
     export openmpvv_compile_flags="${openmpvv_compile_flags} -fopenmp-version=45"
  fi
  perform_make OMP_VERSION="${testcase_omp_version}" SOURCES="${testcase}"
  rc=$?
  echo
  echo "DONE:  Single OPENMP_VV test case: ${testcase}"
  echo "       Source file:  ${testcase_file_path}"
  echo "       make rc: ${rc}"
  if [ -f "${openmpvv_dir}/bin/${testcase}.o" ]; then
     echo "       Binary ${openmpvv_dir}/bin/${testcase}.o exists!"
     echo "       If compile worked, you may rerun the binary with this command:"
     echo " ${openmpvv_dir}/bin/${testcase}.o"
  else
     echo "       Expected binary ${openmpvv_dir}/bin/${testcase}.o does NOT exist!"
  fi
  echo
  popd || exit
  exit ${rc}
fi

# Run test-suite.
if [ "${make_target}" == "all" ]; then
  if [ "${SKIP_OPENMPVV_45}" != 1 ]; then
    # Run OpenMP 4.5 tests
    echo "--------------------------- START OMP 4.5 TESTING ---------------------"
    perform_make OMP_VERSION=4.5
    echo
    echo "--------------------------- OMP 4.5 Detailed Results ---------------------------" >> combined-results.txt
    echo "--------------------------- OMP 4.5 Results ---------------------------" > abrev.combined-results.txt
    make_OpenMP_VV_reports 45
  fi

  if [ "${SKIP_OPENMPVV_50}" != 1 ]; then
    enable_xnack=0
    if gpu_needs_xnack_for_usm "${AOMP_GPU}" && ! is_apu && [ "${HSA_XNACK}" == "" ]; then
      export HSA_XNACK=1
      enable_xnack=1
      echo "Turning on HSA_XNACK=1 for 5.0 to allow USM tests to pass."
    fi
    # Run OpenMP 5.0 tests.
    echo "--------------------------- START OMP 5.0 TESTING ---------------------"
    export openmpvv_compile_flags="${openmpvv_compile_flags} -fopenmp-version=50"
    perform_make OMP_VERSION=5.0 SOURCES="${openmpvv_test_filter}"
    echo
    echo "--------------------------- OMP 5.0 Detailed Results ---------------------------" >> combined-results.txt
    echo "--------------------------- OMP 5.0 Results ---------------------------" >> abrev.combined-results.txt
    make_OpenMP_VV_reports 50
    if [ "${enable_xnack}" == 1 ]; then
      unset HSA_XNACK
    fi
  fi

  if [ "${SKIP_OPENMPVV_51}" != 1 ]; then
    enable_xnack=0
    if gpu_needs_xnack_for_usm "${AOMP_GPU}" && ! is_apu && [ "${HSA_XNACK}" == "" ]; then
      export HSA_XNACK=1
      enable_xnack=1
      echo "Turning on HSA_XNACK=1 for 5.1 to allow USM tests to pass."
    fi
    echo "--------------------------- START OMP 5.1 TESTING ---------------------"
    # Run OpenMP 5.1 tests
    export openmpvv_compile_flags="${openmpvv_compile_flags} -fopenmp-version=51"
    perform_make OMP_VERSION=5.1 SOURCES="${openmpvv_test_filter}"
    echo
    echo "--------------------------- OMP 5.1 Detailed Results ---------------------------" >> combined-results.txt
    echo "--------------------------- OMP 5.1 Results ---------------------------" >> abrev.combined-results.txt
    make_OpenMP_VV_reports 51
    openmpvv_test_filter=""
    if [ "${enable_xnack}" == 1 ]; then
      unset HSA_XNACK
    fi
  fi

  if [ "${SKIP_OPENMPVV_52}" != 1 ]; then
    enable_xnack=0
    if gpu_needs_xnack_for_usm "${AOMP_GPU}" && ! is_apu && [ "${HSA_XNACK}" == "" ]; then
      export HSA_XNACK=1
      enable_xnack=1
      echo "Turning on HSA_XNACK=1 for 5.2 to allow USM tests to pass."
    fi
    echo "--------------------------- START OMP 5.2 TESTING ---------------------"
    # Run OpenMP 5.2 tests
    export openmpvv_compile_flags="${openmpvv_compile_flags} -fopenmp-version=52"
    perform_make OMP_VERSION=5.2 SOURCES="${openmpvv_test_filter}"
    echo
    echo "--------------------------- OMP 5.2 Detailed Results ---------------------------" >> combined-results.txt
    echo "--------------------------- OMP 5.2 Results ---------------------------" >> abrev.combined-results.txt
    make_OpenMP_VV_reports 52
    if [ "${enable_xnack}" == 1 ]; then
      unset HSA_XNACK
    fi
  fi
fi

echo "========================= ALL TESTING COMPLETE ! ====================="
echo

cat combined-results.txt
pwd
echo
echo
echo
cat abrev.combined-results.txt
echo
popd > /dev/null || exit
