#!/usr/bin/env bash
set -x
CKRepoURL='https://github.com/ROCm/composable_kernel.git'
CKRepoBranchName='develop'
CKBenchmarkRepoBranchName='main'

# We grab the total system memory and assume a requirement of 10GB per process
# when building CK. This is likely a bit conservative.
CKBuildParallelism=$(free -g | grep Mem | awk '{print int($2/10)}')

realpath=$(realpath $0)
thisdir=$(dirname $realpath)

. $thisdir/aomp_common_vars

export PATH=$AOMP/bin:$PATH

function printHelp {
  set +x
  echo "Usage: run_composable-kernels.sh"
  echo "  -h: Show this help message"
  echo "  -i: Install the (incremental) CK build"
  echo "  -r: Rebuild the CK repo"
  echo "  -u: Update the CK repo"
  echo "  -b: Update the CK benchmarks repo"
  echo "  -s <suite>: Select <suite> from:"                  \
       "[benchmarks client-examples smoke regression skip]. (Default: skip)"
  echo "  -t <test>: Run <test> from selected suite (e.g. 'gemm/fa1.yaml')"
  exit 0
}

# Given a target GPU architecture, this returns a list of available GPU indices.
function getIndexListByTargetArch {
  TargetArch=$1
  if [ -z "${TargetArch}" ]; then
    echo "Error: No target architecture was provided"
    return 1
  fi

  # Build list of target arch indices
  # First, get a detailed list of all available and visible GPUs.
  # Then match lines containing the target arch and capture the bracketed index.
  # Finally, we use xargs to format the results into a tidy, single-line list.
  OnlyVisibleDevices=""
  if [ ! -z "${ROCR_VISIBLE_DEVICES}" ]; then
    OnlyVisibleDevices="$(echo "-d ${ROCR_VISIBLE_DEVICES}" | tr ',' ' ')"
  fi
  GPUList="$(rocm-smi --showproductname ${OnlyVisibleDevices})"
  GPURegex="s/^GPU\[([0-9]+)\].*${TargetArch}$/\1/p"
  TargetArchIndexList=$(echo "${GPUList}" | sed -En "${GPURegex}" | xargs echo)

  # Return the space-separated index list string (e.g. "0 1 3 4")
  echo "${TargetArchIndexList}"
}

# Given an array of commands to execute, distribute them onto multiple GPUs.
function distributeWorkToGPUs {
  # Capture array argument
  WorkItems=("$@")

  # Sanity checks
  if [ "${#WorkItems[@]}" -eq 0 ] ||
    ([ "${#WorkItems[@]}" -eq 1 ] && [ -z "${WorkItems[@]}" ]); then
    echo "Error: Received empty list of commands"
    exit 1
  fi

  # Get and count available GPUs (as list)
  GPUList="$(getIndexListByTargetArch "${CK_GPU_TARGETS}")"
  GPUCount=$(echo "${GPUList}" | wc -w)
  if [ ${GPUCount} -le 0 ]; then
    echo "No target GPUs available"
    exit 1
  fi

  # Make the GPU index list available within the test sub shells
  export GPUList

  # Run the work items, using GNU parallel
  # Note: {%} provides the job slot (thread) index, {#} the job sequence index
  echo "Running ${#WorkItems[@]} work items in parallel, using ${GPUCount} GPUs"
  parallel -j ${GPUCount} --line-buffer \
    ' read -ra AvailableGPUs <<< "${GPUList}"
      GPUIndex=$(({%} - 1))
      SelectedGPU=${AvailableGPUs[GPUIndex]}
      echo "[GPU ${SelectedGPU}, JOB {#}] Running: {}"
      export ROCR_VISIBLE_DEVICES=${SelectedGPU}
      # Execute the actual work item
      bash -c {}
      ReturnCode=$?
      echo "[GPU ${SelectedGPU}, JOB {#}] Finished with exit code ${ReturnCode}"
      exit ${ReturnCode}
    ' ::: "${WorkItems[@]}"

  # Check the overall exit status of parallel
  ParallelExitCode=$?
  if [ ${ParallelExitCode} -eq 0 ]; then
    echo "All tests completed successfully."
  elif [ ${ParallelExitCode} -eq 255 ]; then
    echo "One or more tests failed (Parallel was signaled to stop)."
    exit 1
  else
    # GNU Parallel exit codes 1-100 indicate number of failed jobs
    echo "Warning: ${ParallelExitCode} tests failed."
    exit 1
  fi
}

# Return an export command, prefixing current LD_LIBRARY_PATH with AOMP_LIB_PATH
function getLDLibraryPathExportCmd {
  # Make sure that we prefer the AOMP libraries over the system ones
  LDLibraryPath="${AOMP_LIB_PATH}"
  if [ -n "${LD_LIBRARY_PATH}" ]; then
    LDLibraryPath="${LDLibraryPath}:${LD_LIBRARY_PATH}"
  fi
  echo "export LD_LIBRARY_PATH=${LDLibraryPath}"
}

# Some tests may require an installed instance of CK.
ShouldInstallCK='no'
# For some situations during testing it may not be desired to rebuild the CK repo.
ShouldRebuildCK='no'
# While doing perf / other compiler work, keeping CK fix is useful.
ShouldUpdateCKRepo='no'

# CK Benchmarks is priate, maybe do not want to update it.
ShouldUpdateCKBenchmarks='no'

# CK may be run using different test- or benchmark-suites.
SelectedSuite='skip'

# CK may be run using a specfic test from the selected suite.
SelectedTest=''

while getopts "hirubs:t:" opt; do
  case ${opt} in
  h)
    printHelp
    ;;
  i)
    # Install the CK build
    ShouldInstallCK='yes'
    ;;
  r)
    # Rebuild the CK repo
    ShouldRebuildCK='yes'
    ;;
  u)
    # Update the CK repo
    ShouldUpdateCKRepo='yes'
    ;;
  b)
    # Update the CK benchmarks repo
    ShouldUpdateCKBenchmarks='yes'
    ;;
  s)
    # Select benchmark or test suite.
    # To support this as an optional argument, we take a look at the next.
    case ${OPTARG} in
      skip)
        # Skip running any suite.
        SelectedSuite='skip'
        ;;
      benchmarks)
        # Run the CK benchmarks.
        SelectedSuite="${OPTARG}"
        ;;
      client-examples)
        # Build and run the client examples provided by CK.
        SelectedSuite="${OPTARG}"
        ;;
      examples)
        # Build and run the examples provided by CK.
        SelectedSuite="${OPTARG}"
        ;;
      smoke)
        # A minimal smoke test suite.
        SelectedSuite="${OPTARG}"
        ;;
      regression)
        # A minimal regression test suite.
        SelectedSuite="${OPTARG}"
        ;;
      *)
        # If there's a following string which does not start with '-'
        # we interpret it as an attempt at providing an unknown suite.
        if [[ "${OPTARG}" =~ ^[^-].*$ ]]; then
          echo "Unknown suite: ${OPTARG}"
          printHelp
        fi
        ;;
    esac
    ;;
  t)
    SelectedTest="${OPTARG}"
    ;;
  *)
    echo "Unknown option: -$opt"
    exit 1
    ;;
  esac
done

# Set the default build prefix, i.e., build-top-level
: ${CK_TOP:=$AOMP_REPOS_TEST/composable-kernels}
: ${CK_REPO:=$CK_TOP/ck-src}
: ${CK_BUILD:=$CK_TOP/ck-build}
: ${CK_BENCHMARK_REPO:=$CK_TOP/ck-benchmark}
# Move this to its own place, to avoid potential permission conflicts with certain setups.
: ${CK_BENCHMARK_RESULT:=$CK_TOP/ck-benchmark-result}
: ${CK_INSTALL:=$CK_TOP/ck-install}
: ${CK_CLIENT_EXAMPLES_SOURCE:=$CK_REPO/client_example}
: ${CK_CLIENT_EXAMPLES_BUILD:=$CK_TOP/ck-client-examples-build}
# Run regular and client examples on multiple GPUs (if present)
: ${CK_EXAMPLES_PARALLEL:='yes'}
: ${CK_EXAMPLES_PREFIX:='example_'}
: ${CK_EXAMPLES_LOG_LOCATION:=$CK_TOP/ck-examples-logs}
: ${CK_TESTS_LOG_LOCATION:=$CK_TOP/ck-tests-logs}

# Some client-examples may take long, override this to skip tests
# e.g. CK_CLIENT_EXAMPLES_TO_EXCLUDE=("10_grouped_convnd_bwd_data" "24_grouped_conv_activation")
: ${CK_CLIENT_EXAMPLES_TO_EXCLUDE:=""}

# Get some info on the system
: ${ROCM_PATH:=/opt/rocm}
: ${CK_GPU_TARGETS:=''}
: ${AOMP_LIB_PATH:="${AOMP}/.."}

if [ -z ${CK_GPU_TARGETS} ]; then
  NumGpuArchs=$(amdgpu-arch | sort | uniq | wc -l)
  if [ ${NumGpuArchs} -gt 1 ]; then
    echo "Error: More than one GPU architecture detected. This may cause issues."
    echo "       Please set the CK_GPU_TARGETS variable to the desired GPU arch."
    exit 1
  else
    # If only one GPU arch is detected, set it as the default.
    CK_GPU_TARGETS=$(amdgpu-arch | uniq)
  fi
fi

echo "Building for ${CK_GPU_TARGETS}"

# Check if user overrode number of parallel build jobs
if [ ! -z ${CK_BUILD_PARALLELISM} ]; then
  CKBuildParallelism=${CK_BUILD_PARALLELISM}
fi

if [ ! -d ${CK_TOP} ]; then
  mkdir -p ${CK_TOP} || exit 1
fi

if [ ! -d ${CK_REPO} ]; then
  git clone ${CKRepoURL} ${CK_REPO}
elif [ "${ShouldUpdateCKRepo}" == 'yes' ]; then
  pushd ${CK_REPO} || exit 1
  git reset --hard origin/${CKRepoBranchName}
  git pull
  # TODO: Write current SHA to somewhere such that it is known which SHA
  #       was tested in this nightly run.
  popd
fi

CKBuildTool='make'
if command -v ninja >/dev/null ; then
  CmakeGenerator="-GNinja"
  CKBuildTool='ninja'
fi

# TODO Fix / Finalize the cmake command
CKCmakeCmd="cmake ${CmakeGenerator} -B ${CK_BUILD} -S ${CK_REPO} -DCMAKE_PREFIX_PATH=${ROCM_PATH} -DCMAKE_INSTALL_PREFIX=${CK_INSTALL} "
CKCmakeCmd+="-DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ -DCMAKE_HIP_COMPILER=${AOMP}/bin/clang++ "
CKCmakeCmd+="-DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=${CK_GPU_TARGETS} "
# For some reason, CK on gfx12 wants this set.
CKCmakeCmd+="-DBUILD_DEV=On"

# Ensure CK build directory is cleaned.
if [ "${ShouldRebuildCK}" == 'yes' ]; then
  echo "Rebuilding the CK repo w/ ${CKBuildParallelism} parallel jobs."
  rm -rf ${CK_BUILD} || exit 1

  echo "CMake Config Command:"
  echo "${CKCmakeCmd}"

  ${CKCmakeCmd}
  if [ $? -ne 0 ]; then
    exit 1
  fi
fi

# Ensure CK install directory is cleaned.
if [ "${ShouldInstallCK}" == 'yes' ]; then
  echo "Purging previous CK installation directory."
  rm -rf ${CK_INSTALL} || exit 1
fi

# Perform (incremental) CK build
if [ "${ShouldRebuildCK}" == 'yes' ] || [ "${ShouldInstallCK}" == 'yes' ]; then
  pushd ${CK_BUILD} || exit 1

  /usr/bin/time -o build-times.tlog ${CKBuildTool} -j ${CKBuildParallelism}
  if [ $? -ne 0 ]; then
    exit 1
  fi

  popd
fi

# Perform CK installation
if [ "${ShouldInstallCK}" == 'yes' ]; then
  pushd ${CK_BUILD} || exit 1

  # TODO: Check parallelism. This may use all available threads.
  /usr/bin/time -o install-times.tlog ${CKBuildTool} install
  if [ $? -ne 0 ]; then
    exit 1
  fi

  popd
fi

echo "Run suite: ${SelectedSuite}"

# Check if parallel execution is requested and possible
UseParallel=0
if ([ "${SelectedSuite}" == 'client-examples' ] ||
    [ "${SelectedSuite}" == 'examples' ]) &&
    [ "${CK_EXAMPLES_PARALLEL}" == 'yes' ]; then
  if [ ! -z "$(command -v parallel)" ]; then
    UseParallel=1
  else
    echo "Warning: Parallel execution requested, but 'parallel' is not available"
  fi
fi

if [ "${SelectedSuite}" == 'smoke' ]; then
  echo "Running CK smoke tests"
  if [ ! -d "${CK_TESTS_LOG_LOCATION}" ]; then
    mkdir -p "${CK_TESTS_LOG_LOCATION}" || exit 1
  fi
  pushd ${CK_BUILD} || exit 1
  ${CKBuildTool} -j 16 smoke 2>&1 | tee "${CK_TESTS_LOG_LOCATION}/smoke_tests.log"
  echo "Log at ${CK_TESTS_LOG_LOCATION}/smoke_tests.log"
  popd
fi

if [ "${SelectedSuite}" == 'regression' ]; then
  echo "Running CK regression tests"
  if [ ! -d "${CK_TESTS_LOG_LOCATION}" ]; then
    mkdir -p "${CK_TESTS_LOG_LOCATION}" || exit 1
  fi
  pushd ${CK_BUILD} || exit 1
  ${CKBuildTool} -j 16 regression 2>&1 | tee "${CK_TESTS_LOG_LOCATION}/regression_tests.log"
  echo "Log at ${CK_TESTS_LOG_LOCATION}/regression_tests.log"
  popd
fi

# Handle CK benchmarks (also as default, if no suite has been explicitly selected)
if [ "${SelectedSuite}" == 'benchmarks' ]; then
  # The CK benchmarks repo appears to be private (for the time being).

  if [ ! -d ${CK_BENCHMARK_REPO} ]; then
    echo "CK Benchmarks repo not found. This is a private repo."
    echo "Please clone with your preferred method into ${CK_BENCHMARK_REPO}"
    exit 1
  elif [ "${ShouldUpdateCKBenchmarks}" == 'yes' ]; then
    pushd ${CK_BENCHMARK_REPO} || exit 1
    git reset --hard origin/${CKBenchmarkRepoBranchName}
    git pull
    # TODO: Dump SHA somewhere
    popd
  fi

  if [ ! -d ${CK_BENCHMARK_RESULT} ]; then
    mkdir -p ${CK_BENCHMARK_RESULT} || exit 1
  fi

  # Check if a specific test was requested
  # If yes: check if it exists
  if [ ! -z ${SelectedTest} ]; then
    if [ ! -f "${CK_BENCHMARK_REPO}/benchmarks/${SelectedTest}" ]; then
      echo "Error: Selected benchmark does not exist:"
      echo "       ${CK_BENCHMARK_REPO}/benchmarks/${SelectedTest}"
      exit 1
    fi
    echo "Selected benchmark: ${CK_BENCHMARK_REPO}/benchmarks/${SelectedTest}"
  else
    # Default benchmark
    SelectedTest="gemm/fa1.yaml"
  fi

  # This is the command. It requires the envar CK_PROFILER_DIR to be set to the directory
  # in the CK build tree that contains the CkProfiler binary.
  CKBenchmarkTest="../benchmarks/${SelectedTest}"
  CKBenchmarkName=$(basename ${CKBenchmarkTest})
  CKBenchmarkResultOutput="${CK_BENCHMARK_RESULT}/${CKBenchmarkName}.output"
  CKBenchmarkBackend='ck'
  CKBenchmarkCmd="./run_gemm.py ${CKBenchmarkBackend} ${CKBenchmarkTest} --output ${CKBenchmarkResultOutput}"
  CKBenchmarkProfilerExport="export CK_PROFILER_DIR=${CK_BUILD}/bin"
  CKBenchmarkLDLibraryPathExport=$(getLDLibraryPathExportCmd)

  pushd ${CK_BENCHMARK_REPO}/scripts || exit 1

  echo "Benchmark Command: ${CKBenchmarkProfilerExport} ; ${CKBenchmarkCmd}"
  ${CKBenchmarkLDLibraryPathExport}
  ${CKBenchmarkProfilerExport}
  ${CKBenchmarkCmd}

  popd

  echo "Benchmark Output File: ${CKBenchmarkResultOutput}"
fi

# Handle CK client examples
if [ "${SelectedSuite}" == 'client-examples' ]; then
  # Configure and build the client examples
  CKCmakeCmd="cmake ${CmakeGenerator} "
  CKCmakeCmd+="-B ${CK_CLIENT_EXAMPLES_BUILD} -S ${CK_CLIENT_EXAMPLES_SOURCE} "
  CKCmakeCmd+="-DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ "
  CKCmakeCmd+="-DCMAKE_HIP_COMPILER=${AOMP}/bin/clang++ "
  CKCmakeCmd+="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
  CKCmakeCmd+="-DCMAKE_PREFIX_PATH=${AOMP_LIB_PATH}/cmake;${CK_INSTALL} "
  CKCmakeCmd+="-DGPU_TARGETS=${CK_GPU_TARGETS} "

  CKClientExLDLibraryPathExport=$(getLDLibraryPathExportCmd)

  if [ "${ShouldInstallCK}" != 'yes' ]; then
    echo "Warning: client-examples selected without required CK installation."
    echo "         Please, make sure CK is properly installed."
  fi

  echo "Rebuilding the CK client-examples"
  rm -rf ${CK_CLIENT_EXAMPLES_BUILD} || exit 1

  echo "CMake Config Command:"
  echo "${CKCmakeCmd}"

  ${CKCmakeCmd}
  if [ $? -ne 0 ]; then
    exit 1
  fi

  pushd ${CK_CLIENT_EXAMPLES_BUILD} || exit 1

  ${CKBuildTool}
  if [ $? -ne 0 ]; then
    exit 1
  fi

  # Process directories to exclude
  # Usage of here-string to avoid sub-shell and removal of potential parentheses
  read -ra DirsToExclude <<< "${CK_CLIENT_EXAMPLES_TO_EXCLUDE//[()]/}"

  # Build argument list for find
  # If globbed directories are provided, the list is expanded correspondingly
  # Hence, the argument list can become quite large and we count the excluded
  # directories while traversing the resulting argument (path) list
  NumExcludedDirs=0
  FindArgs=(. -mindepth 1 -maxdepth 2 -type d \()
  for ExcludedDir in "${DirsToExclude[@]}"; do
    FindArgs+=(-path "./${ExcludedDir}" -o)
    echo "Excluding client-examples: ./${ExcludedDir}"
    ((++NumExcludedDirs))
  done
  echo "Excluded ${NumExcludedDirs} client-example directories"
  # Also, we always want to prune "./CMakeFiles" from the results
  FindArgs+=(-path "*CMakeFiles*" \) -prune -o)
  # If requested, filter the selected tests
  if [ ! -z ${SelectedTest} ]; then
    echo "Filtering client-example paths: ./${SelectedTest}"
    FindArgs+=(-path "./${SelectedTest}")
  fi
  # Finally, we want to print the remaining retrieved executables
  FindArgs+=(-type f -executable -print)

  # Gather client-example executables
  ExamplesToRun=$(find "${FindArgs[@]}" | sort)

  # Build run command list
  # Note: Usage of here-string to avoid sub-shell
  declare -a ExampleRunCmds
  while read -r ExamplePath; do
    # Sanity check
    if [ -z "${ExamplePath}" ]; then
      continue
    fi
    # Get directory and basename part, then construct the log file path
    ExampleDir=$(dirname "${ExamplePath}")
    ExampleName=$(basename "${ExamplePath}")
    ExampleLogfile="${ExampleDir}/run_${ExampleName}.log"
    # Construct and add the example run command with tee
    RunCmd="echo \"Running client-example: ${ExamplePath}\";"
    RunCmd+="\"${ExamplePath}\" | tee \"${ExampleLogfile}\""
    ExampleRunCmds+=("${RunCmd}")
  done <<< "${ExamplesToRun}"

  NumJobs=${#ExampleRunCmds[@]}
  echo "Found ${NumJobs} client-examples to run"
  if [ ${NumJobs} == 0 ]; then
    # When running this script, we should expect to run something
    # Exit silently, but indicate error via returncode
    exit 1
  fi

  # Prepare library path
  ${CKClientExLDLibraryPathExport}

  # Run each client-example
  if [ ${UseParallel} == 1 ]; then
    # Parallel execution, using multiple GPUs
    distributeWorkToGPUs "${ExampleRunCmds[@]}"
  else
    # Sequential execution, using a single (default) GPU
    # Use 'bash -c' since simple string does not work
    echo "Running client-examples sequentially, using a single GPU"
    for RunCmd in "${ExampleRunCmds[@]}"; do
      bash -c "${RunCmd}"
    done
  fi

  popd
fi

# Handle CK's regular examples
if [ "${SelectedSuite}" == 'examples' ]; then
  # CK's examples require a CK build to be present
  if [ ! -d "${CK_BUILD}" ]; then
    echo "Error: Missing CK build directory: ${CK_BUILD}"
    exit 1
  fi

  # Build argument list for find
  FindArgs=(. -mindepth 1 -maxdepth 1 -type f)
  if [ -z ${SelectedTest} ]; then
    # No filtering requested: select all "*" tests
    SelectedTest="*"
  else
    # Communicate filter to user
    echo "Filtering examples: ${SelectedTest}"
  fi
  FindArgs+=(-name "${CK_EXAMPLES_PREFIX}${SelectedTest}")
  FindArgs+=(-executable -print)

  # Gather executables
  pushd "${CK_BUILD}/bin" || exit 1
  ExamplesToRun=$(find "${FindArgs[@]}" | sort)

  # Build run command list
  # Note: Usage of here-string to avoid sub-shell
  declare -a ExampleRunCmds
  while read -r ExamplePath; do
    # Sanity check
    if [ -z "${ExamplePath}" ]; then
      continue
    fi
    # Construct the log file path
    ExampleName=$(basename "${ExamplePath}")
    ExampleLogfile="${CK_EXAMPLES_LOG_LOCATION}/run_${ExampleName}.log"
    # Construct and add the example run command with tee
    RunCmd="echo \"Running example: ${ExamplePath}\";"
    RunCmd+="\"${ExamplePath}\" | tee \"${ExampleLogfile}\""
    ExampleRunCmds+=("${RunCmd}")
  done <<< "${ExamplesToRun}"

  NumJobs=${#ExampleRunCmds[@]}
  echo "Found ${NumJobs} examples to run"
  if [ ${NumJobs} == 0 ]; then
    # When running this script, we should expect to run something
    # Exit silently, but indicate error via returncode
    exit 1
  fi

  # Avoid picking up stale logs
  echo "Purging CK examples logs"
  rm -rf ${CK_EXAMPLES_LOG_LOCATION} || exit 1
  mkdir -p ${CK_EXAMPLES_LOG_LOCATION} || exit 1

  # Run each example
  if [ ${UseParallel} == 1 ]; then
    # Parallel execution, using multiple GPUs
    distributeWorkToGPUs "${ExampleRunCmds[@]}"
  else
    # Sequential execution, using a single (default) GPU
    # Use 'bash -c' since simple string does not work
    echo "Running examples sequentially, using a single GPU"
    for RunCmd in "${ExampleRunCmds[@]}"; do
      bash -c "${RunCmd}"
    done
  fi

  popd
fi
