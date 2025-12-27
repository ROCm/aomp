#!/bin/bash
# Test with OMP_DEFAULT_DEVICE=2 and OMP_TARGET_OFFLOAD=DISABLED
# Before the fix, this would crash with "device number '2' out of range"
# After the fix, omp_get_default_device() should return 0 (the initial device)
export OMP_DEFAULT_DEVICE=2
export OMP_TARGET_OFFLOAD=DISABLED
./$1


