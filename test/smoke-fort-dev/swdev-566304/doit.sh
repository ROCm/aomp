set -x
HSA_XNACK=0 numactl -N0 ./rush_larsen_gpu_omp_fort 1000 20 2>&1  | tee out0 
HSA_XNACK=1 numactl -N0 ./rush_larsen_gpu_omp_fort 1000 20 2>&1  | tee out1
grep STATS out0
grep STATS out1

