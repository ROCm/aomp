#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
#
# run_gpurun_rocprof_env.sh : compare environments with different process launchers
#                            
# Rule1: Always run rocprof, rocgdb binaries from gpurun and not the opposite.
# Rule2: When using mpirun, always run it first, followed by gpurun, followed by rocXXX, followed by the application binary
#
AOMP=${AOMP:-/opt/rocm}
_gpurun_opts="-m -v" 
echo 
echo "=====1==== env  >  env_standalone.out"
env | sort >env_standalone.out
echo
echo "=====2==== $AOMP/lib/llvm/bin/gpurun $_gpurun_opts env  >  env_gpurun.out"
"$AOMP"/lib/llvm/bin/gpurun $_gpurun_opts env | sort >env_gpurun.out

echo
echo "=====3==== $AOMP/lib/llvm/bin/gpurun $_gpurun_opts $AOMP/bin/rocprofv3 --kernel-trace --hsa-trace --memory-copy-trace --stats -- env  >  env_gpurun_rocprof.out"
"$AOMP"/lib/llvm/bin/gpurun $_gpurun_opts "$AOMP"/bin/rocprofv3 --kernel-trace --hsa-trace --memory-copy-trace --stats -- env | sort >env_gpurun_rocprof.out

echo
echo "=====4==== mpirun -np 1 $AOMP/lib/llvm/bin/gpurun $_gpurun_opts $AOMP/bin/rocprofv3 --kernel-trace --hsa-trace --memory-copy-trace --stats -- env  >  env_mpirun_gpurun_rocprof.out"
mpirun -np 1 "$AOMP"/lib/llvm/bin/gpurun $_gpurun_opts "$AOMP"/bin/rocprofv3 --kernel-trace --hsa-trace --memory-copy-trace --stats -- env | sort >env_mpirun_gpurun_rocprof.out

echo
echo ---------- diff 1 env_standalone.out     2 env_gpurun.out
diff env_standalone.out env_gpurun.out

echo
echo ---------- diff 2 env_gpurun.out         3 env_gpurun_rocprof.out
diff env_gpurun.out env_gpurun_rocprof.out

echo
echo ---------- diff 3 env_gpurun_rocprof.out 4 env_mpirun_gpurun_rocprof.out
diff env_gpurun_rocprof.out env_mpirun_gpurun_rocprof.out
