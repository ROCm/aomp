#!/bin/bash
#
# run_qe.sh:
#   Build Quantum ESPRESSO (PWscf) with the AOMP OpenMP-offload toolchain
#   (amdflang) and run the PW test-suite on 4 MPI ranks (one GPU per rank).
#
# The QE OpenMP-5 offload sources are cloned into $AOMP_REPOS_TEST by
# clone_test.sh (see manifests/test_<version>.xml). Source repo:
#   https://gitlab.com/QEF/q-e-omp-repository  (branch develop_omp5_75omp)
#
# Usage:
#   ./run_qe.sh              configure + build pw.x + run the PW test-suite
#   ./run_qe.sh nocmake      skip ./configure (and veryclean), rebuild + test
#   ./run_qe.sh rerun        skip build, just re-run the PW test-suite
#
# This is a single self-contained file: the OpenMPI affinity helpers are
# embedded as shell functions below, and the per-rank GPU pinning launcher
# (gpu_affinity_close.sh) is generated on the fly into the QE test-suite
# directory at run time and removed afterwards.
#

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
export AOMP_USE_CCACHE=0

# shellcheck disable=SC1091
. "$thisdir"/aomp_common_vars
# --- end standard header ----

# Offload kernel tracing is very noisy across a full test-suite; default off.
export LIBOMPTARGET_KERNEL_TRACE=${LIBOMPTARGET_KERNEL_TRACE:-0}

# Setup AOMP variables
AOMP=${AOMP:-$HOME/rocm/srock/llvm}

# Use function to set and test AOMP_GPU
setaompgpu

# --- QE specific variables (all overridable from the environment) ----------

# Directory name the QE fork is cloned into (matches manifest 'path=' attribute).
AOMP_QE_REPO_NAME=${AOMP_QE_REPO_NAME:-q-e-omp}
QE_REPO=${QE_REPO:-$AOMP_REPOS_TEST/$AOMP_QE_REPO_NAME}

# ROCm root passed to QE configure (--with-rocm). ROCM_PATH can be set
# explicitly; or computed relative to the given $AOMP llvm directory.
ROCM_PATH=${ROCM_PATH:-"$(realpath -m "$(realpath -m "$AOMP")/../..")"}

# GPU arch passed to QE configure (--with-gpu-arch). Defaults to detected AOMP_GPU.
GPU_ARCH=${GPU_ARCH:-$AOMP_GPU}

# GPU-aware OpenMPI, built as an AOMP supplemental component (build_supp.sh).
OPENMPI_INSTALL=${OPENMPI_INSTALL:-$AOMP_SUPP/openmpi}

# Make amdflang, amdclang and mpifort/mpicc/mpirun discoverable.
export PATH=$OPENMPI_INSTALL/bin:$AOMP/bin:$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$OPENMPI_INSTALL/lib:$AOMP/lib:$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH

# clang-linker-wrapper / lld resolve -l libraries via LIBRARY_PATH in addition
# to the -L flags. QE hardcodes '-L$(ROCM_PATH)/lib', so add the ROCm lib dirs
# here too (covering lib64) to fix 'unable to find library -lamdhip64' when the
# HIP runtime lives in a non-default location.
export LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$AOMP/lib:$LIBRARY_PATH

# QE's 'ARCH=amdflang' path drives compilation through the OpenMPI wrapper
# 'mpifort' and decides the compiler flavor from 'mpifort --version'. If the
# OpenMPI installation was built against gfortran, configure detects gfortran
# and silently falls back to the generic path (wrong DFLAGS, no HIP in
# make.inc). Overriding the wrapped compilers makes 'mpifort' report amdflang
# so the ROCm/HIP OpenMP-offload branch (-D__HIP -D__ROCBLAS -D__OPENMP_GPU,
# hipcc rule, rocfft/rocblas/rocsolver link flags) is taken. No rebuild of
# OpenMPI is required.
export OMPI_FC=${OMPI_FC:-amdflang}
export OMPI_CC=${OMPI_CC:-amdclang}
export OMPI_CXX=${OMPI_CXX:-amdclang++}

echo "ROCM_PATH= $ROCM_PATH"
# --- Runtime environment ---------------------------------------------------
export OMP_NUM_THREADS=1

# HSA / offload runtime tweaks (safe with any OpenMPI).
export HSA_ENABLE_IPC_MODE_LEGACY=1
export HSA_ENABLE_SDMA=1
# Fix for QE hangs (HSA issue?)
export OMPX_FORCE_SYNC_REGIONS=1

# GPU-aware UCX/UCC transport settings (from pw_test_suite.sh). These REQUIRE an
# OpenMPI whose UCX is built with ROCm support. Forcing pml=ucx on a plain
# OpenMPI aborts MPI_Init with "PML ucx cannot be selected", so they are opt-in:
# export QE_USE_UCX=1 only when running against a ROCm-aware UCX OpenMPI.
if [ "${QE_USE_UCX:-0}" == "1" ]; then
  # GPU-aware UCX transport (single node).
  export UCX_IB_GPU_DIRECT_RDMA=no
  export OMPI_MCA_pml="ucx"
  export OMPI_MCA_osc="ucx"
  export UCX_TLS="self,sm,rocm"

  # UCC collectives: GPU-optimized Allreduce (drives RCCL). Big speedup for QE:
  # without it calbec's Allreduce is ~34x slower and the whole SCF is ~4x slower
  # (8m17s -> 1m55s). HCOLL stays off (Mellanox-only, re-inits IB).
  export OMPI_MCA_coll_hcoll_enable=0
  export OMPI_MCA_coll_ucc_enable=1
  export OMPI_MCA_coll_ucc_priority=100

  export UCX_HANDLE_ERRORS="bt"
  export UCX_MAX_RNDV_RAILS=1
fi

# ===========================================================================
# Embedded OpenMPI affinity helpers (from openmpi_helpers.sh)
# ===========================================================================
get_ngpu_node()
{
    local count=0
    local file
    for file in /dev/dri/by-path/pci-*-render; do
	[[ -e ${file} ]] || continue
	[[ -w ${file} ]] && count=$(( count + 1))
    done
    echo "${count}"
}

get_openmpi_node_binding()
{
    if [ "${#}" -ne "5" ]; then
	echo "Usage: ${FUNCNAME[0]} <nnodes> <ntasks> <nthreads> <ntasks_per_node> <ngpus_per_node>"
        return 1
    fi
    local nnodes=${1}
    local ntasks=${2}
    local nthreads=${3}
    local ntasks_per_node=${4}
    local ngpus_per_node=${5}
    if [ "${ngpus_per_node}" -gt "${ntasks_per_node}" ]; then
	ngpus_per_node=${ntasks_per_node}
    fi
    if [ "${ntasks}" -lt "${ntasks_per_node}" ]; then
        ntasks_per_node=${ntasks}
    fi
    local nthread_node_req=$(( ntasks_per_node * nthreads ))
    local nthread_node_avail=$(( $(lscpu --parse=CPU | tail -n1 ) + 1 ))
    local ncpu_node_avail=$(( $(lscpu --parse=CORE | tail -n1 ) + 1 ))
    local nsocket_node_avail=$(( $(lscpu --parse=SOCKET | tail -n1 ) + 1 ))
    local nnuma_node_avail=$(( $(lscpu --parse=NODE | tail -n1 ) + 1 ))
    local ntasks_per_socket=$(( ntasks_per_node / nsocket_node_avail ))
    local ntasks_per_numa=$(( ntasks_per_node / nnuma_node_avail ))

    # Underpopulation rules
    if [ $(( ntasks_per_socket * nsocket_node_avail )) -ne "${ntasks_per_node}" ]; then
        ntasks_per_socket=0
    fi
    if [ $(( ntasks_per_numa * nnuma_node_avail )) -ne "${ntasks_per_node}" ]; then
        ntasks_per_numa=0
    fi

    local task_place=numa
    if [ "${ntasks_per_socket}" -eq 0 ]; then
	ntasks_per_numa=${ntasks_per_node}
	nnuma_node_avail=${nsocket_node_avail}
	task_place=node
    elif [ "${ntasks_per_numa}" -eq 0 ]; then
	ntasks_per_numa=${ntasks_per_socket}
	nnuma_node_avail=${nsocket_node_avail}
	task_place=socket
    fi
    local ngpus_per_numa=$(( ngpus_per_node / nnuma_node_avail ))
    if [ ${ngpus_per_numa} -eq 0 ]; then
	ngpus_per_numa=1
    fi
    local mpibind

    # Set ntasks_per_numa according to GPUs (if any)
    if [ "${ngpus_per_numa}" -gt "${ntasks_per_numa}" ]; then
	ntasks_per_numa=${ngpus_per_numa}
    fi
    if [ "${ntasks_per_numa}" -gt "${ntasks}" ]; then
	ntasks_per_numa=${ntasks}
    fi

    # Distribute numa nodes evenly via nthreads
    ncpus_per_numa=$(( ncpu_node_avail / nnuma_node_avail ))
    nthreads_per_numa=$(( ncpus_per_numa / ntasks_per_numa ))
    if [ "${nthreads_per_numa}" -gt "${nthreads}" ]; then
	nthreads=${nthreads_per_numa}
    fi

    if [ "${use_ht:-false}" == "true" ]; then
	local corespec=hwthread
    fi

    if [ "${nthread_node_req}" -gt "${nthread_node_avail}" ]; then
	# Not enough CPUs available!
	echo "ERROR: nthread_node_req=${nthread_node_req} > nthread_node_avail=${nthread_node_avail}"
	return 1
    elif [ "${nthread_node_req}" -le "${nthread_node_avail}" ] && [ "${nthread_node_req}" -gt "${ncpu_node_avail}" ]; then
	mpibind="ppr:${ntasks_per_node}:node:PE=${nthreads} --bind-to hwthread"
    elif [ "${nthread_node_req}" -le "${ncpu_node_avail}" ]; then
	mpibind="ppr:${ntasks_per_numa}:${task_place}:PE=${nthreads} --bind-to ${corespec:-core}"
    fi
    echo "--map-by ${mpibind}"
}

get_openmpi_host_binding()
{
    if [ "$#" -ne "1" ]; then
	echo "Usage: ${FUNCNAME[0]} <nodelist>"
	return 1
    fi

    local nodelist=${1}
    local ncore_node=$(( $(lscpu --parse=CPU | tail -n1 ) + 1 ))
    local nodes
    mapfile -t nodes < <(scontrol show hostnames "${nodelist}")
    # Construct nodelist with slot counts for OpenMPI
    local nodes_ompi=""
    local node
    for node in "${nodes[@]}"; do
	if [ -z "${nodes_ompi}" ]; then
	    nodes_ompi="${node}:${ncore_node}"
	else
	    nodes_ompi+=",${node}:${ncore_node}"
	fi
    done
    echo "-host ${nodes_ompi}"
}

get_openmpi_env()
{
    local env_ompi=""
    for envvar in $(env | grep 'OMP_\|UCX_\|HSA_\|HIP_\|DBCSR_\|LD_PRELOAD'); do
	env_ompi+="-x ${envvar} "
    done
    echo "${env_ompi}"
}

# ---------------------------------------------------------------------------
# write_gpu_affinity_launcher <path>
#   Materialize the per-rank GPU pinning launcher (gpu_affinity_close.sh).
#   It is exec'd once per MPI rank by mpirun; it computes ROCR_VISIBLE_DEVICES
#   from the local rank and then runs the command passed as arguments (pw.x).
# ---------------------------------------------------------------------------
write_gpu_affinity_launcher()
{
  local out=${1}
  cat > "${out}" <<'GPUAFF'
#!/bin/bash

function count_gpus()
{
    local count=0
    local file
    for file in $(ls /dev/dri/by-path/pci-*-render 2> /dev/null); do
	[[ -w ${file} ]] && count=$(( count + 1))
    done
    echo ${count}
}

function get_gpu_vendor_name()
{
    local vendorid
    local vendorname="unknown"

    # Find vendor name of first accessible render entry
    vendorid=0x0
    for render in $(find /sys/class/drm/ -maxdepth 1 -regex '.*/renderD1[0-9][0-9]+'); do
	if [ -f ${render}/device/vendor ]; then
	    vendorid=$(<${render}/device/vendor)
	    break
	fi
    done

    case ${vendorid} in
	0x8086)
	    vendorname="intel"
	    ;;
	0x1002|0x1022)
	    vendorname="amd"
	    ;;
	0x10de|0x10DE)
	    vendorname="nvidia"
	    ;;
    esac
    echo ${vendorname}
}

if [ ! -z "${OMPI_COMM_WORLD_SIZE:-}" ]; then
    mpi_size=${OMPI_COMM_WORLD_SIZE}
    mpi_rank=${OMPI_COMM_WORLD_RANK}
    mpi_node_size=${OMPI_COMM_WORLD_LOCAL_SIZE}
    mpi_node_rank=${OMPI_COMM_WORLD_LOCAL_RANK}
else
    mpi_size=${PMI_SIZE:-"-1"}
    mpi_rank=${PMI_RANK:-"-1"}
    mpi_node_size=${MPI_LOCALNRANKS:-1}
    mpi_node_rank=${MPI_LOCALRANKID:-0}
fi
gpus_per_node=${gpus_per_node:-$(count_gpus)}

if [ ! -z "${gpu_rank_ids}" ]; then
    gpu_id_map=( ${gpu_rank_ids} )
    gpu_id_map_explicit="true"

    # Sanity check
    if [ "${#gpu_id_map[*]}" -ne "${mpi_node_size}" ]; then
	echo "ERROR: gpu_id_map size does not match MPI node size"
	exit 1
    fi

else
    # gpu_id_map=( $(seq 0 1 ${gpus_per_node}) )
    gpu_id_map_explicit="false"
fi

if [ "${gpus_per_node}" -eq "0" ]; then
    VISIBLE_DEVICES=""
else
    if [ ${gpu_id_map_explicit} == "false" ]; then
	gpus_per_rank=$((gpus_per_node / mpi_node_size ))
	if [ ${gpus_per_rank} -gt 1 -a ${use_mgpu:-"false"} == "true" ]; then
	    devid=$((gpus_per_rank*mpi_node_rank))
	    VISIBLE_DEVICES=${devid}
	    for (( devid=devid+1; devid<(mpi_node_rank+1)*gpus_per_rank; devid++)); do
		VISIBLE_DEVICES="${VISIBLE_DEVICES},${devid}"
	    done
	else
	    if [ ${gpus_per_node} -le ${mpi_node_size} ]; then
		ranks_per_gpu=$((mpi_node_size/gpus_per_node))
	    else
		ranks_per_gpu=1
	    fi
	    devid=$(( mpi_node_rank / ranks_per_gpu ))

	    VISIBLE_DEVICES=${devid}
	fi
    else
	# Apply user provided explicit mapping per rank
	VISIBLE_DEVICES=${gpu_id_map[${mpi_node_rank}]}
    fi
fi

gpu_vendor=$(get_gpu_vendor_name)

case ${gpu_vendor} in
    amd)
	export ROCR_VISIBLE_DEVICES=${VISIBLE_DEVICES}
	# export HIP_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}
	# export GPU_DEVICE_ORDINAL=${ROCR_VISIBLE_DEVICES}
	echo "host=$(hostname) ngpus=${gpus_per_node} nprocs=${mpi_size}, rank=${mpi_rank}, nprocs_local=${mpi_node_size}, rank_local=${mpi_node_rank} ROCR_VISIBLE_DEVICES=${VISIBLE_DEVICES}"
	;;
    nvidia)
	export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}
	echo "host=$(hostname) ngpus=${gpus_per_node} nprocs=${mpi_size}, rank=${mpi_rank}, nprocs_local=${mpi_node_size}, rank_local=${mpi_node_rank} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
	;;
    intel)
	export ZE_AFFINITY_MASK=${VISIBLE_DEVICES}
	echo "host=$(hostname) ngpus=${gpus_per_node} nprocs=${mpi_size}, rank=${mpi_rank}, nprocs_local=${mpi_node_size}, rank_local=${mpi_node_rank} ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"
	;;
    *)
	echo "host=$(hostname) ngpus=${gpus_per_node} nprocs=${mpi_size}, rank=${mpi_rank}, nprocs_local=${mpi_node_size}, rank_local=${mpi_node_rank} Unknown GPU vendor"
	;;
esac

ulimit -s unlimited
"${@}"
GPUAFF
  chmod 755 "${out}"
}

if [ ! -d "$QE_REPO" ]; then
  echo "ERROR: QE sources not found at $QE_REPO"
  echo "       Run clone_test.sh first (needs a manifest entry for the QE fork)."
  exit 1
fi

ulimit -s unlimited

# --- run_pw_tests: modify run-pw.sh, run PW test-suite, restore ------------
run_pw_tests() {
  cd "$QE_REPO"/test-suite || exit 1

  # Generate the per-rank GPU pinning launcher used by mpirun.
  gpu_affinity="$QE_REPO/test-suite/gpu_affinity_close.sh"
  write_gpu_affinity_launcher "$gpu_affinity"

  # Gather OpenMPI affinity/binding. SLURM vars are used when present;
  # otherwise sensible single-node defaults (4 ranks) are applied.
  nnodes=${SLURM_NNODES:-1}
  ntasks=${SLURM_NTASKS:-4}
  ntasks_per_node=${SLURM_NTASKS_PER_NODE:-$(( ntasks / nnodes ))}
  nthreads=${OMP_NUM_THREADS:-1}
  ngpus_per_node="$(get_ngpu_node)"
  # -host binding needs SLURM (scontrol); skip it for a direct single-node run.
  if [ -n "$SLURM_NODELIST" ]; then
    ompi_host=$(get_openmpi_host_binding "$SLURM_NODELIST")
  else
    ompi_host=""
  fi
  ompi_env="$(get_openmpi_env)"
  map_binding=$(get_openmpi_node_binding "$nnodes" "$ntasks" "$nthreads" "$ntasks_per_node" "$ngpus_per_node")

  echo "=== OpenMPI configuration ==="
  echo "Nodes: ${nnodes} | Tasks: ${ntasks} | Tasks/node: ${ntasks_per_node} | Threads: ${nthreads} | GPUs/node: ${ngpus_per_node}"
  echo "OpenMPI host binding: ${ompi_host}"
  echo "Map binding: ${map_binding}"
  echo "OpenMPI environment: ${ompi_env}"
  echo "=== End OpenMPI configuration ==="

  # Inject the affinity-aware launcher into run-pw.sh. The ompi_* variables are
  # left literal here and expanded at run time by run-pw.sh (exported below);
  # the gpu_affinity_close.sh path is expanded now.
  sed -i 's|export PARA_PREFIX=.*|export PARA_PREFIX="mpirun ${ompi_env} -n ${ntasks} ${ompi_host} ${map_binding} --report-bindings '"${gpu_affinity}"' "|' run-pw.sh
  chmod 770 run-pw.sh

  export ompi_env ntasks ompi_host map_binding

  VERBOSE=${VERBOSE:-"1"}
  set -x
  if [ "$VERBOSE" -eq 0 ]; then
    make run-tests-pw NPROCS=4 2>&1 | tee log_aware.log > /dev/null
  else
    make run-tests-pw NPROCS=4 2>&1 | tee log_aware.log
  fi
  ret=${PIPESTATUS[0]}
  set +x

  # Restore run-pw.sh to its original launcher and clean up the generated helper.
  sed -i 's|export PARA_PREFIX=.*|export PARA_PREFIX="mpirun -np $QE_USE_MPI"|' run-pw.sh
  chmod 770 run-pw.sh
  rm -f "$gpu_affinity"

  if [ "$ret" -ne 0 ]; then
    echo "quantum-espresso-pw" >> "$QE_REPO"/failing-tests.txt
  else
    echo "quantum-espresso-pw" >> "$QE_REPO"/passing-tests.txt
  fi
  return "$ret"
}

# --- rerun: skip the build, just re-run the PW tests -----------------------
if [ "$1" == "rerun" ]; then
  run_pw_tests
  exit $?
fi

cd "$QE_REPO" || exit 1
rm -f make-fail.txt failing-tests.txt passing-tests.txt

# --- Configure -------------------------------------------------------------
if [ "$1" != "nocmake" ]; then
  # Full from-scratch clean so compilation is always exercised with the current
  # compiler. 'make veryclean' runs 'make clean' (all object files/executables)
  # and additionally removes the generated make.inc and the bundled external
  # libraries (lapack, FoX, MBD, wannier90, devxlib) plus install/config.*.
  echo "make veryclean"
  make veryclean || true

  echo "./configure ARCH=amdflang --enable-omp_gpu --with-rocm=$ROCM_PATH --enable-omp_mpi_gpu --with-gpu-arch=$GPU_ARCH"
  ./configure ARCH=amdflang \
    --enable-omp_gpu \
    --with-rocm="$ROCM_PATH" \
    --enable-omp_mpi_gpu \
    --with-gpu-arch="$GPU_ARCH"
  ret=$?
  if [ $ret -ne 0 ]; then
    echo "quantum-espresso-configure" >> make-fail.txt
    exit 1
  fi
fi

# --- Build PWscf -----------------------------------------------------------
make -j"$AOMP_JOB_THREADS" pw
ret=$?
if [ $ret -ne 0 ]; then
  echo "quantum-espresso-pw" >> make-fail.txt
  exit 1
fi

# --- Run the PW test-suite -------------------------------------------------
run_pw_tests
exit $?
