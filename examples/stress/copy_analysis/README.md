copyAnalysis - mesaure hipMemcpy performance under different allocations.
=======================================================

To compare XNACK-enabled vs XNACK-disabled, build with:
make
and run with:
HSA_XNACK=0 ./copyAnalysis
and
HSA_XNACK=1 ./copyAnalysis

The test outputs .csv files for each of three configurations:
Configuration 1: malloc, hipMalloc, hipMemcpy
Configuration 2: malloc, hipMemAdvise, hipMemcpy
Configuration 3: malloc, hipMemPrefetchAsync, hipDeviceSynchronize, hipMemcpy

Configuration 3 is currently faulty in that it applies prefetching to a non device pointer, which seems to result in a no op.

Output files are marked with xnack-enabled or xnack-disabled, based on run configuration.
The test assumes the node has been booted in XNACK-disabled mode.