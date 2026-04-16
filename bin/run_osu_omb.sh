#!/bin/bash
#
# run_osu_omb.sh: Script to build and run OSU Micro-Benchmarks for
#                 GPU-Aware MPI testing with ROCm on AMD GPUs.
#
# This script:
#   1. Clones/downloads OSU Micro-Benchmarks if not present
#   2. Builds OMB with ROCm-aware MPI (from buildrocmopenmpi)
#   3. Runs comprehensive GPU-aware MPI benchmarks
#

realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
export AOMP_USE_CCACHE=0

. "$thisdir/aomp_common_vars"

trap 'echo ""; echo "Benchmark interrupted by user. Exiting..."; exit 130' INT TERM
set -e

################################################################################
# OSU Micro-Benchmark Configuration
################################################################################

OMB_VERSION=${OMB_VERSION:-7.3}
OMB_URL="https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OMB_VERSION}.tar.gz"
OMB_CNAME="osu-micro-benchmarks"
OMB_BUILD_DIR=${AOMP_SUPP_BUILD}/${OMB_CNAME}
OMB_INSTALL_DIR=${AOMP_SUPP_INSTALL}/${OMB_CNAME}-${OMB_VERSION}
OMB_LINK=${AOMP_SUPP}/${OMB_CNAME}

################################################################################
# ROCm-aware MPI Configuration (from buildrocmopenmpi)
################################################################################

# Derive ROCM_PATH - for AOMP installations, AOMP itself is the ROCm root
function derive_rocm_path(){
  if [ -d "$AOMP/include/hip" ] || [ -d "$AOMP/include/rocm-core" ] ; then
    ROCM_PATH=$AOMP
  elif [ -n "$LLVM_INSTALL_LOC" ] && [ -d "$LLVM_INSTALL_LOC/../../../include/hip" ] ; then
    ROCM_PATH=$(realpath "$LLVM_INSTALL_LOC/../../..")
  elif [ -d "$AOMP/../include/hip" ] ; then
    ROCM_PATH=$(realpath "$AOMP/..")
  else
    echo "Error: Cannot determine ROCM_PATH."
    echo "       Expected ROCm headers at \$AOMP/include/hip or similar."
    echo "       AOMP=$AOMP"
    exit 1
  fi
  ROCM_PATH=$(realpath "$ROCM_PATH")
}

derive_rocm_path

# MPI Installation from buildrocmopenmpi
ROCM_OPENMPI_DIR=${AOMP_SUPP}/rocmopenmpi
if [ ! -d "$ROCM_OPENMPI_DIR" ] ; then
  echo "Error: ROCm-aware OpenMPI not found at $ROCM_OPENMPI_DIR"
  echo "       Please run: build_supp.sh rocmopenmpi"
  exit 1
fi

# UCX Installation
UCX_DIR=${AOMP_SUPP}/ucx
if [ ! -d "$UCX_DIR" ] ; then
  echo "Error: UCX not found at $UCX_DIR"
  echo "       Please run: build_supp.sh ucx"
  exit 1
fi

# Set environment for MPI
export PATH=${ROCM_OPENMPI_DIR}/bin:${AOMP}/bin:${PATH}
export LD_LIBRARY_PATH=${ROCM_OPENMPI_DIR}/lib:${UCX_DIR}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH}
export MPI_HOME=${ROCM_OPENMPI_DIR}
export MPI_INCLUDE=${MPI_HOME}/include
export MPI_LIBDIR=${MPI_HOME}/lib

# Compiler settings for MPI
export OMPI_CC=${AOMP}/bin/hipcc
export OMPI_CXX=${AOMP}/bin/hipcc

################################################################################
# Benchmark Runtime Configuration
################################################################################

# Results directory
RESULTS_DIR=${RESULTS_DIR:-${AOMP_SUPP_BUILD}/omb-results}

# Benchmark paths (set after install)
OMB_DIR=""
PT2PT_DIR=""
COLLECTIVE_DIR=""
ONESIDED_DIR=""

# Test parameters
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_GPUS=${NUM_GPUS:-4}
ITERATIONS=${ITERATIONS:-100}
WARMUP=${WARMUP:-10}

# Message sizes for different test types
MSG_SIZE_SMALL="1:4096"          # For latency tests
MSG_SIZE_LARGE="1:$((16*1024*1024))"  # For bandwidth tests
MSG_SIZE_FULL="1:$((16*1024*1024))"   # Full range

################################################################################
# MPI Run Configuration for GPU-Aware MPI
################################################################################

# Base MPI options for GPU-aware communication
MPI_BASE_OPTS=""
MPI_BASE_OPTS+=" --mca pml ucx"
MPI_BASE_OPTS+=" --mca osc ucx"
MPI_BASE_OPTS+=" --mca btl ^vader,tcp,openib,uct"
MPI_BASE_OPTS+=" --mca coll_hcoll_enable 0"

# Process mapping - use flexible slot-based mapping with no strict binding
MPI_BIND_OPTS=""
MPI_BIND_OPTS+=" --bind-to none"
MPI_BIND_OPTS+=" --map-by slot"

# UCX options for ROCm GPU memory
MPI_UCX_OPTS=""
MPI_UCX_OPTS+=" -x UCX_TLS=self,sm,rocm_copy,rocm_ipc"
MPI_UCX_OPTS+=" -x UCX_MEMTYPE_CACHE=y"
MPI_UCX_OPTS+=" -x UCX_ROCM_COPY_D2H_MODE=auto"
MPI_UCX_OPTS+=" -x UCX_ROCM_COPY_H2D_MODE=auto"
MPI_UCX_OPTS+=" -x UCX_LOG_LEVEL=warn"
MPI_UCX_OPTS+=" -x UCX_ROCM_COPY_SIGPOOL_MAX_ELEMS=32768"

# Combine all MPI options
MPI_GPU_OPTS="$MPI_BASE_OPTS $MPI_BIND_OPTS $MPI_UCX_OPTS"

# Simplified options for quick tests
MPI_QUICK_OPTS="$MPI_BASE_OPTS --bind-to none --map-by slot -x UCX_TLS=self,sm,rocm_copy,rocm_ipc -x UCX_MEMTYPE_CACHE=y -x UCX_LOG_LEVEL=warn -x UCX_ROCM_COPY_SIGPOOL_MAX_ELEMS=32768"

################################################################################
# Build Functions
################################################################################

function check_dependencies(){
  echo "Checking dependencies..."
  
  # Check for ROCm-aware OpenMPI
  if [ ! -d "$ROCM_OPENMPI_DIR" ] ; then
    echo "Error: ROCm-aware OpenMPI not found at $ROCM_OPENMPI_DIR"
    echo "       Please run: $thisdir/build_supp.sh rocmopenmpi"
    return 1
  fi
  
  # Check for mpicc
  if ! command -v mpicc &> /dev/null ; then
    echo "Error: mpicc not found in PATH"
    echo "       PATH=$PATH"
    return 1
  fi
  
  # Check for hipcc
  if [ ! -x "$AOMP/bin/hipcc" ] ; then
    echo "Error: hipcc not found at $AOMP/bin/hipcc"
    return 1
  fi
  
  echo "  ROCm-aware OpenMPI: $ROCM_OPENMPI_DIR"
  echo "  UCX: $UCX_DIR"
  echo "  ROCM_PATH: $ROCM_PATH"
  echo "  hipcc: $AOMP/bin/hipcc"
  echo "  mpicc: $(which mpicc)"
  return 0
}

function download_omb(){
  echo "Downloading OSU Micro-Benchmarks ${OMB_VERSION}..."
  
  if [ -d "$OMB_BUILD_DIR" ] ; then
    echo "  Removing existing build directory..."
    rm -rf "$OMB_BUILD_DIR"
  fi
  
  mkdir -p "$OMB_BUILD_DIR"
  cd "$OMB_BUILD_DIR"
  
  echo "  Downloading from $OMB_URL"
  wget -q "$OMB_URL" -O "osu-micro-benchmarks-${OMB_VERSION}.tar.gz"
  
  echo "  Extracting..."
  tar -xzf "osu-micro-benchmarks-${OMB_VERSION}.tar.gz"
  
  echo "  Download complete."
}

function build_omb(){
  echo "Building OSU Micro-Benchmarks ${OMB_VERSION} with ROCm support..."
  
  local _srcdir="$OMB_BUILD_DIR/osu-micro-benchmarks-${OMB_VERSION}"
  
  if [ ! -d "$_srcdir" ] ; then
    echo "Error: Source directory not found: $_srcdir"
    echo "       Please run with 'download' option first."
    return 1
  fi
  
  cd "$_srcdir"
  
  # Clean previous install if exists
  if [ -d "$OMB_INSTALL_DIR" ] ; then
    echo "  Removing previous installation..."
    rm -rf "$OMB_INSTALL_DIR"
  fi
  mkdir -p "$OMB_INSTALL_DIR"
  
  echo "  Configuring with ROCm support..."
  echo "    CC=$(which mpicc)"
  echo "    CXX=$(which mpicxx)"
  echo "    ROCM_PATH=$ROCM_PATH"
  echo "    Install prefix=$OMB_INSTALL_DIR"
  
  ./configure --prefix="$OMB_INSTALL_DIR" \
              CC="$(which mpicc)" \
              CXX="$(which mpicxx)" \
              CPPFLAGS="-D__HIP_PLATFORM_AMD__=1" \
              --enable-rocm \
              --with-rocm="$ROCM_PATH"
  
  echo "  Building with ${AOMP_JOB_THREADS} parallel jobs..."
  make -j"${AOMP_JOB_THREADS}"
  
  echo "  Installing..."
  make install
  
  # Create symbolic link
  if [ -L "$OMB_LINK" ] ; then
    rm "$OMB_LINK"
  fi
  ln -sfr "$OMB_INSTALL_DIR" "$OMB_LINK"
  
  echo "  Build complete."
  echo "  Installed to: $OMB_INSTALL_DIR"
  echo "  Symbolic link: $OMB_LINK"
}

function set_benchmark_paths(){
  # Set paths to benchmark binaries
  if [ -d "$OMB_LINK/libexec/osu-micro-benchmarks/mpi" ] ; then
    OMB_DIR="$OMB_LINK/libexec/osu-micro-benchmarks/mpi"
  elif [ -d "$OMB_INSTALL_DIR/libexec/osu-micro-benchmarks/mpi" ] ; then
    OMB_DIR="$OMB_INSTALL_DIR/libexec/osu-micro-benchmarks/mpi"
  else
    echo "Error: OMB installation not found."
    echo "       Expected at $OMB_LINK/libexec/osu-micro-benchmarks/mpi"
    echo "       Please run: $0 build"
    return 1
  fi
  
  PT2PT_DIR="$OMB_DIR/pt2pt"
  COLLECTIVE_DIR="$OMB_DIR/collective"
  ONESIDED_DIR="$OMB_DIR/one-sided"
  
  # Verify directories exist
  if [ ! -d "$PT2PT_DIR" ] ; then
    echo "Error: Point-to-point benchmark directory not found: $PT2PT_DIR"
    return 1
  fi
  
  return 0
}

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

print_subheader() {
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "$1"
    echo "--------------------------------------------------------------------------------"
}

log_system_info() {
    local log_file=$1
    echo "System Information:" | tee -a "$log_file"
    echo "  Timestamp: $(date)" | tee -a "$log_file"
    echo "  Hostname: $(hostname)" | tee -a "$log_file"
    echo "  AOMP: $AOMP" | tee -a "$log_file"
    echo "  ROCM_PATH: $ROCM_PATH" | tee -a "$log_file"
    echo "  MPI: $MPI_HOME" | tee -a "$log_file"
    echo "  UCX: $UCX_DIR" | tee -a "$log_file"
    echo "  OMB: $OMB_DIR" | tee -a "$log_file"
    echo "  MPI GPU Options: $MPI_GPU_OPTS" | tee -a "$log_file"
    "${AOMP}/bin/rocminfo" 2>/dev/null | grep -E "Name:|Marketing Name:" | head -8 | tee -a "$log_file" || true
    echo "" | tee -a "$log_file"
}

run_benchmark() {
    local benchmark=$1
    local name=$2
    local nprocs=$3
    local args=$4
    local log_file=$5
    local engine=$6
    local mpi_opts=${7:-$MPI_GPU_OPTS}

    echo "" | tee -a "$log_file"
    print_subheader "$name ($engine)" | tee -a "$log_file"
    
    local full_cmd="mpirun -n $nprocs $mpi_opts $benchmark $args"
    echo "Command: $full_cmd" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    
    if eval "$full_cmd" 2>&1 | tee -a "$log_file"; then
        echo "[PASS] $name completed successfully" | tee -a "$log_file"
    else
        echo "[FAIL] $name failed" | tee -a "$log_file"
    fi
}

################################################################################
# Point-to-Point Benchmarks
################################################################################

run_pt2pt_benchmarks() {
    local engine=$1
    local sdma_setting=$2
    local gpus=$3
    local log_file=$4

    export HSA_ENABLE_SDMA=$sdma_setting
    export HIP_VISIBLE_DEVICES=$gpus

    print_header "Point-to-Point Benchmarks ($engine)" | tee -a "$log_file"
    echo "HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA" | tee -a "$log_file"
    echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" | tee -a "$log_file"

    # Latency Test (D-D)
    run_benchmark "$PT2PT_DIR/osu_latency" \
        "Latency (Device-to-Device)" 2 \
        "-d rocm -m $MSG_SIZE_SMALL D D" \
        "$log_file" "$engine"

    # Latency Test (H-D)
    run_benchmark "$PT2PT_DIR/osu_latency" \
        "Latency (Host-to-Device)" 2 \
        "-d rocm -m $MSG_SIZE_SMALL H D" \
        "$log_file" "$engine"

    # Latency Test (D-H)
    run_benchmark "$PT2PT_DIR/osu_latency" \
        "Latency (Device-to-Host)" 2 \
        "-d rocm -m $MSG_SIZE_SMALL D H" \
        "$log_file" "$engine"

    # Bandwidth Test (D-D)
    run_benchmark "$PT2PT_DIR/osu_bw" \
        "Bandwidth (Device-to-Device)" 2 \
        "-d rocm -m $MSG_SIZE_LARGE D D" \
        "$log_file" "$engine"

    # Bandwidth Test (H-D)
    run_benchmark "$PT2PT_DIR/osu_bw" \
        "Bandwidth (Host-to-Device)" 2 \
        "-d rocm -m $MSG_SIZE_LARGE H D" \
        "$log_file" "$engine"

    # Bandwidth Test (D-H)
    run_benchmark "$PT2PT_DIR/osu_bw" \
        "Bandwidth (Device-to-Host)" 2 \
        "-d rocm -m $MSG_SIZE_LARGE D H" \
        "$log_file" "$engine"

    # Bidirectional Bandwidth Test (D-D)
    run_benchmark "$PT2PT_DIR/osu_bibw" \
        "Bidirectional Bandwidth (Device-to-Device)" 2 \
        "-d rocm -m $MSG_SIZE_LARGE D D" \
        "$log_file" "$engine"

    # Multiple Bandwidth / Message Rate Test (D-D)
    run_benchmark "$PT2PT_DIR/osu_mbw_mr" \
        "Multiple Bandwidth/Message Rate (Device-to-Device)" 2 \
        "-d rocm -m $MSG_SIZE_LARGE D D" \
        "$log_file" "$engine"

    # Multi-pair Latency Test (D-D)
    run_benchmark "$PT2PT_DIR/osu_multi_lat" \
        "Multi-pair Latency (Device-to-Device)" 2 \
        "-d rocm -m $MSG_SIZE_SMALL D D" \
        "$log_file" "$engine"
}

################################################################################
# Collective Benchmarks
################################################################################

run_collective_benchmarks() {
    local engine=$1
    local sdma_setting=$2
    local gpus=$3
    local log_file=$4
    local nprocs=$5

    export HSA_ENABLE_SDMA=$sdma_setting
    export HIP_VISIBLE_DEVICES=$gpus

    print_header "Collective Benchmarks ($engine) - $nprocs processes" | tee -a "$log_file"
    echo "HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA" | tee -a "$log_file"
    echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" | tee -a "$log_file"

    run_benchmark "$COLLECTIVE_DIR/osu_allreduce" \
        "MPI_Allreduce" "$nprocs" \
        "-d rocm -m $MSG_SIZE_FULL" \
        "$log_file" "$engine"

    run_benchmark "$COLLECTIVE_DIR/osu_allgather" \
        "MPI_Allgather" "$nprocs" \
        "-d rocm -m $MSG_SIZE_FULL" \
        "$log_file" "$engine"

    run_benchmark "$COLLECTIVE_DIR/osu_alltoall" \
        "MPI_Alltoall" "$nprocs" \
        "-d rocm -m $MSG_SIZE_FULL" \
        "$log_file" "$engine"

    run_benchmark "$COLLECTIVE_DIR/osu_bcast" \
        "MPI_Bcast" "$nprocs" \
        "-d rocm -m $MSG_SIZE_FULL" \
        "$log_file" "$engine"

    run_benchmark "$COLLECTIVE_DIR/osu_reduce" \
        "MPI_Reduce" "$nprocs" \
        "-d rocm -m $MSG_SIZE_FULL" \
        "$log_file" "$engine"

    run_benchmark "$COLLECTIVE_DIR/osu_gather" \
        "MPI_Gather" "$nprocs" \
        "-d rocm -m $MSG_SIZE_FULL" \
        "$log_file" "$engine"

    run_benchmark "$COLLECTIVE_DIR/osu_scatter" \
        "MPI_Scatter" "$nprocs" \
        "-d rocm -m $MSG_SIZE_FULL" \
        "$log_file" "$engine"

    run_benchmark "$COLLECTIVE_DIR/osu_reduce_scatter" \
        "MPI_Reduce_scatter" "$nprocs" \
        "-d rocm -m $MSG_SIZE_FULL" \
        "$log_file" "$engine"

    run_benchmark "$COLLECTIVE_DIR/osu_barrier" \
        "MPI_Barrier" "$nprocs" \
        "" \
        "$log_file" "$engine"
}

################################################################################
# One-Sided (RMA) Benchmarks
################################################################################

run_onesided_benchmarks() {
    local engine=$1
    local sdma_setting=$2
    local gpus=$3
    local log_file=$4

    export HSA_ENABLE_SDMA=$sdma_setting
    export HIP_VISIBLE_DEVICES=$gpus

    print_header "One-Sided (RMA) Benchmarks ($engine)" | tee -a "$log_file"
    echo "HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA" | tee -a "$log_file"
    echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" | tee -a "$log_file"

    run_benchmark "$ONESIDED_DIR/osu_put_latency" \
        "Put Latency (Device-to-Device)" 2 \
        "-d rocm -m $MSG_SIZE_SMALL D D" \
        "$log_file" "$engine"

    run_benchmark "$ONESIDED_DIR/osu_put_bw" \
        "Put Bandwidth (Device-to-Device)" 2 \
        "-d rocm -m $MSG_SIZE_LARGE D D" \
        "$log_file" "$engine"

    run_benchmark "$ONESIDED_DIR/osu_put_bibw" \
        "Put Bidirectional Bandwidth (Device-to-Device)" 2 \
        "-d rocm -m $MSG_SIZE_LARGE D D" \
        "$log_file" "$engine"

    run_benchmark "$ONESIDED_DIR/osu_get_latency" \
        "Get Latency (Device-to-Device)" 2 \
        "-d rocm -m $MSG_SIZE_SMALL D D" \
        "$log_file" "$engine"

    run_benchmark "$ONESIDED_DIR/osu_get_bw" \
        "Get Bandwidth (Device-to-Device)" 2 \
        "-d rocm -m $MSG_SIZE_LARGE D D" \
        "$log_file" "$engine"

    run_benchmark "$ONESIDED_DIR/osu_acc_latency" \
        "Accumulate Latency" 2 \
        "-d rocm -m $MSG_SIZE_SMALL" \
        "$log_file" "$engine"
}

################################################################################
# Quick Sanity Test
################################################################################

run_quick_test() {
    set_benchmark_paths || return 1
    
    mkdir -p "$RESULTS_DIR"
    local log_file=$RESULTS_DIR/quick_test_$TIMESTAMP.log
    
    print_header "Quick Sanity Test" | tee -a "$log_file"
    log_system_info "$log_file"

    export HSA_ENABLE_SDMA=1
    export HIP_VISIBLE_DEVICES=0,1
    
    echo "Testing SDMA engine..." | tee -a "$log_file"
    run_benchmark "$PT2PT_DIR/osu_latency" \
        "Quick Latency Test" 2 \
        "-d rocm -m 1:1024 D D" \
        "$log_file" "SDMA" "$MPI_QUICK_OPTS"
    
    run_benchmark "$PT2PT_DIR/osu_bw" \
        "Quick Bandwidth Test" 2 \
        "-d rocm -m $((1024*1024)):$((4*1024*1024)) D D" \
        "$log_file" "SDMA" "$MPI_QUICK_OPTS"

    export HSA_ENABLE_SDMA=0
    export HIP_VISIBLE_DEVICES=2,3
    
    echo "Testing Blit Kernel engine..." | tee -a "$log_file"
    run_benchmark "$PT2PT_DIR/osu_latency" \
        "Quick Latency Test" 2 \
        "-d rocm -m 1:1024 D D" \
        "$log_file" "BlitKernel" "$MPI_QUICK_OPTS"
    
    run_benchmark "$PT2PT_DIR/osu_bw" \
        "Quick Bandwidth Test" 2 \
        "-d rocm -m $((1024*1024)):$((4*1024*1024)) D D" \
        "$log_file" "BlitKernel" "$MPI_QUICK_OPTS"

    print_header "Quick Test Complete"
    echo "Results saved to: $log_file"
}

################################################################################
# Full Benchmark Suite
################################################################################

run_full_suite() {
    set_benchmark_paths || return 1
    
    print_header "OSU Micro-Benchmarks - GPU-Aware MPI Full Suite"
    echo "Comparing: SDMA vs Blit Kernel copy engines"
    echo "Results directory: $RESULTS_DIR"
    echo "MPI Options: $MPI_GPU_OPTS"
    echo ""

    mkdir -p "$RESULTS_DIR"

    local sdma_log=$RESULTS_DIR/full_suite_SDMA_$TIMESTAMP.log
    log_system_info "$sdma_log"
    
    run_pt2pt_benchmarks "SDMA" 1 "0,1" "$sdma_log"
    run_collective_benchmarks "SDMA" 1 "0,1" "$sdma_log" 2
    run_onesided_benchmarks "SDMA" 1 "0,1" "$sdma_log"

    local blit_log=$RESULTS_DIR/full_suite_BlitKernel_$TIMESTAMP.log
    log_system_info "$blit_log"
    
    run_pt2pt_benchmarks "BlitKernel" 0 "2,3" "$blit_log"
    run_collective_benchmarks "BlitKernel" 0 "2,3" "$blit_log" 2
    run_onesided_benchmarks "BlitKernel" 0 "2,3" "$blit_log"

    local multi_log=$RESULTS_DIR/full_suite_4GPU_$TIMESTAMP.log
    log_system_info "$multi_log"
    
    print_header "4-GPU Collective Benchmarks" | tee -a "$multi_log"
    run_collective_benchmarks "SDMA-4GPU" 1 "0,1,2,3" "$multi_log" 4
    run_collective_benchmarks "BlitKernel-4GPU" 0 "0,1,2,3" "$multi_log" 4

    print_header "Full Suite Complete"
    echo "Results saved to:"
    echo "  - $sdma_log"
    echo "  - $blit_log"
    echo "  - $multi_log"
}

################################################################################
# Bandwidth Focus Test
################################################################################

run_bandwidth_test() {
    set_benchmark_paths || return 1
    
    mkdir -p "$RESULTS_DIR"
    local log_file=$RESULTS_DIR/bandwidth_test_$TIMESTAMP.log
    
    print_header "Bandwidth-Focused Benchmark" | tee -a "$log_file"
    log_system_info "$log_file"

    local large_msg="$((4*1024*1024)):$((16*1024*1024))"

    export HSA_ENABLE_SDMA=1
    export HIP_VISIBLE_DEVICES=0,1
    
    run_benchmark "$PT2PT_DIR/osu_bw" \
        "Bandwidth D-D" 2 "-d rocm -m $large_msg D D" "$log_file" "SDMA"
    run_benchmark "$PT2PT_DIR/osu_bibw" \
        "Bidirectional Bandwidth D-D" 2 "-d rocm -m $large_msg D D" "$log_file" "SDMA"
    run_benchmark "$PT2PT_DIR/osu_mbw_mr" \
        "Multiple BW/MR D-D" 2 "-d rocm -m $large_msg D D" "$log_file" "SDMA"

    export HSA_ENABLE_SDMA=0
    export HIP_VISIBLE_DEVICES=2,3
    
    run_benchmark "$PT2PT_DIR/osu_bw" \
        "Bandwidth D-D" 2 "-d rocm -m $large_msg D D" "$log_file" "BlitKernel"
    run_benchmark "$PT2PT_DIR/osu_bibw" \
        "Bidirectional Bandwidth D-D" 2 "-d rocm -m $large_msg D D" "$log_file" "BlitKernel"
    run_benchmark "$PT2PT_DIR/osu_mbw_mr" \
        "Multiple BW/MR D-D" 2 "-d rocm -m $large_msg D D" "$log_file" "BlitKernel"

    print_header "Bandwidth Test Complete"
    echo "Results saved to: $log_file"
}

################################################################################
# Latency Focus Test
################################################################################

run_latency_test() {
    set_benchmark_paths || return 1
    
    mkdir -p "$RESULTS_DIR"
    local log_file=$RESULTS_DIR/latency_test_$TIMESTAMP.log
    
    print_header "Latency-Focused Benchmark" | tee -a "$log_file"
    log_system_info "$log_file"

    local small_msg="1:4096"

    export HSA_ENABLE_SDMA=1
    export HIP_VISIBLE_DEVICES=0,1
    
    run_benchmark "$PT2PT_DIR/osu_latency" \
        "Latency D-D" 2 "-d rocm -m $small_msg D D" "$log_file" "SDMA"
    run_benchmark "$PT2PT_DIR/osu_latency" \
        "Latency H-D" 2 "-d rocm -m $small_msg H D" "$log_file" "SDMA"
    run_benchmark "$PT2PT_DIR/osu_latency" \
        "Latency D-H" 2 "-d rocm -m $small_msg D H" "$log_file" "SDMA"
    run_benchmark "$PT2PT_DIR/osu_multi_lat" \
        "Multi-pair Latency D-D" 2 "-d rocm -m $small_msg D D" "$log_file" "SDMA"

    export HSA_ENABLE_SDMA=0
    export HIP_VISIBLE_DEVICES=2,3
    
    run_benchmark "$PT2PT_DIR/osu_latency" \
        "Latency D-D" 2 "-d rocm -m $small_msg D D" "$log_file" "BlitKernel"
    run_benchmark "$PT2PT_DIR/osu_latency" \
        "Latency H-D" 2 "-d rocm -m $small_msg H D" "$log_file" "BlitKernel"
    run_benchmark "$PT2PT_DIR/osu_latency" \
        "Latency D-H" 2 "-d rocm -m $small_msg D H" "$log_file" "BlitKernel"
    run_benchmark "$PT2PT_DIR/osu_multi_lat" \
        "Multi-pair Latency D-D" 2 "-d rocm -m $small_msg D D" "$log_file" "BlitKernel"

    print_header "Latency Test Complete"
    echo "Results saved to: $log_file"
}

################################################################################
# Collective Focus Test
################################################################################

run_collective_test() {
    set_benchmark_paths || return 1
    
    mkdir -p "$RESULTS_DIR"
    local log_file=$RESULTS_DIR/collective_test_$TIMESTAMP.log
    
    print_header "Collective Operations Benchmark" | tee -a "$log_file"
    log_system_info "$log_file"

    run_collective_benchmarks "SDMA" 1 "0,1" "$log_file" 2
    run_collective_benchmarks "BlitKernel" 0 "2,3" "$log_file" 2
    run_collective_benchmarks "SDMA-4GPU" 1 "0,1,2,3" "$log_file" 4
    run_collective_benchmarks "BlitKernel-4GPU" 0 "0,1,2,3" "$log_file" 4

    print_header "Collective Test Complete"
    echo "Results saved to: $log_file"
}

################################################################################
# Usage
################################################################################

usage() {
    cat << EOF
Usage: $0 [OPTION]

OSU Micro-Benchmarks for GPU-Aware MPI with ROCm

Build Options:
  download    Download OSU Micro-Benchmarks tarball
  build       Build OSU Micro-Benchmarks with ROCm support
  install     Download and build (full installation)

Benchmark Options:
  quick       Run quick sanity tests (default)
  full        Run full benchmark suite
  bandwidth   Run bandwidth-focused benchmarks
  latency     Run latency-focused benchmarks
  collective  Run collective operation benchmarks

Other Options:
  info        Show configuration information
  help        Show this help message

Environment Variables:
  AOMP              AOMP installation directory (default: \$HOME/rocm/aomp)
  AOMP_SUPP         Supplemental components directory (default: \$HOME/local)
  OMB_VERSION       OSU Micro-Benchmarks version (default: 7.3)
  RESULTS_DIR       Directory for benchmark results
  NUM_GPUS          Number of GPUs to use (default: 4)

Prerequisites:
  Run build_supp.sh to install required components:
    $thisdir/build_supp.sh rocmopenmpi

Examples:
  $0 install      # Download and build OMB
  $0 quick        # Run quick validation
  $0 full         # Run complete benchmark suite
  $0 bandwidth    # Focus on bandwidth tests

Directory Structure:
  Build:   $OMB_BUILD_DIR
  Install: $OMB_INSTALL_DIR
  Link:    $OMB_LINK
  Results: $RESULTS_DIR

EOF
}

show_info() {
    print_header "OSU Micro-Benchmarks Configuration"
    echo "AOMP Installation:"
    echo "  AOMP=$AOMP"
    echo "  ROCM_PATH=$ROCM_PATH"
    echo ""
    echo "MPI Configuration:"
    echo "  ROCM_OPENMPI_DIR=$ROCM_OPENMPI_DIR"
    echo "  UCX_DIR=$UCX_DIR"
    echo "  mpicc=$(which mpicc 2>/dev/null || echo 'not found')"
    echo ""
    echo "OMB Configuration:"
    echo "  OMB_VERSION=$OMB_VERSION"
    echo "  OMB_BUILD_DIR=$OMB_BUILD_DIR"
    echo "  OMB_INSTALL_DIR=$OMB_INSTALL_DIR"
    echo "  OMB_LINK=$OMB_LINK"
    echo ""
    echo "Build Settings:"
    echo "  AOMP_JOB_THREADS=$AOMP_JOB_THREADS"
    echo ""
    echo "Results Directory:"
    echo "  RESULTS_DIR=$RESULTS_DIR"
    echo ""
    
    if [ -L "$OMB_LINK" ] && [ -d "$OMB_LINK" ] ; then
        echo "OMB Status: INSTALLED"
        echo "  Linked to: $(readlink -f "$OMB_LINK")"
    else
        echo "OMB Status: NOT INSTALLED"
        echo "  Run: $0 install"
    fi
}

################################################################################
# Main
################################################################################

case "${1:-quick}" in
    download)
        check_dependencies || exit 1
        download_omb
        ;;
    build)
        check_dependencies || exit 1
        build_omb
        ;;
    install)
        check_dependencies || exit 1
        download_omb
        build_omb
        ;;
    quick)
        check_dependencies || exit 1
        run_quick_test
        ;;
    full)
        check_dependencies || exit 1
        run_full_suite
        ;;
    bandwidth)
        check_dependencies || exit 1
        run_bandwidth_test
        ;;
    latency)
        check_dependencies || exit 1
        run_latency_test
        ;;
    collective)
        check_dependencies || exit 1
        run_collective_test
        ;;
    info)
        show_info
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
esac
