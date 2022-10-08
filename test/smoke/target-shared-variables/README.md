## Impact of deglobalization on LDS

This test shows what happens when the `-mllvm -openmp-opt-disable-deglobalization=true` flag is passed in and deglobalization is disabled.

Without deglobalization the calls to `__kmpc_alloc_shared()` are present in the LLVM IR intermediate code and can be traced all the way into the device assembly where ds_* instructions are present.

When deglobalization is active the calls to `__kmpc_alloc_shared()` are being optimized out.

## LDS Reporting

When enabling `LIBOMPTARGET_KERNEL_TRACE=1` both kernels are reported as no-loop (SGN:4) but the first kernel reports 0B for LDS while the second kernel reports 1540B for LDS. Upon inspecting the device assembly the code emitted for the `__kmpc_alloc_shared()` instruction (in all its locations) looks similar.

```
DEVID: 0 SGN:4 ConstWGSize:256  args: 3 teamsXthrds:(  79X 256) reqd:(   0X   0) lds_usage:0B sgpr_count:9 vgpr_count:14 sgpr_spill_count:0 vgpr_spill_count:0 tripcount:20000 rpc:1 n:__omp_offloading_fd00_26c6a13_main_l56
DEVID: 0 SGN:4 ConstWGSize:256  args: 3 teamsXthrds:(  79X 256) reqd:(   0X   0) lds_usage:1540B sgpr_count:108 vgpr_count:47 sgpr_spill_count:6 vgpr_spill_count:0 tripcount:20000 rpc:1 n:__omp_offloading_fd00_26c6a13_main_l67
```