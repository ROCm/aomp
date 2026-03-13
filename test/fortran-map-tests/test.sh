# !/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier:  MIT

AOMP=${AOMP:-$HOME/rocm/aomp/llvm}
FORTRAN_OMP_MOD_FILES=${FORTRAN_OMP_MOD_FILES:-$HOME/rocm/aomp/lib/llvm/include/flang}
OFFLOAD_RUNTIME_DIR=${OFFLOAD_RUNTIME_DIR:-$HOME/rocm/aomp/lib/llvm/lib}
AOMPHIP=${AOMPHIP:-$(realpath -m $(realpath -m $AOMP)/../..)}
# for ROCm utilities (e.g. rocm_agent_enumerator)
ROCM=${ROCM:-$(realpath -m $(realpath -m $AOMP)/../..)}

echo "compiling basic-exp-map.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  basic-exp-map.f95 -o basic-exp-map.out

echo "compiling basic-example.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  basic-example.f95 -o basic-example.out

echo "compiling basic-example-v2.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  basic-example-v2.f95 -o basic-example-v2.out

echo "compiling basic-double-target-call.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  basic-double-target-call.f95 -o basic-double-target-call.out

echo "compiling even-basicer-example.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  even-basicer-example.f95 -o even-basicer-example.out

echo "compiling main-all.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  main-all.f95 -o main-all.out

echo "compiling main.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  main.f95 -o main.out

echo "compiling no-write-implicit-cap.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  no-write-implicit-cap.f95 -o no-write-implicit-cap.out

echo "compiling milestone-1-map-syntax.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  milestone-1-map-syntax.f95 -o milestone-1-map-syntax.out

echo "compiling milestone-1-map-exact-syntax.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  milestone-1-map-exact-syntax.f95 -o milestone-1-map-exact-syntax.out

echo "compiling decltar-double-target-call.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  decltar-double-target-call.f95 -o decltar-double-target-call.out

echo "compiling constant-index-in-target.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  constant-index-in-target.f95 -o constant-index-in-target.out 

echo "compiling from-to.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  from-to.f95 -o from-to.out

echo "compiling complex.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  complex.f95 -o complex.out

echo "compiling complex-array.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  complex-array.f95 -o complex-array.out

echo "compiling constant-array-access.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  constant-array-access.f95 -o constant-array-access.out

echo "compiling simple-full-struct.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  simple-full-struct.f95 -o simple-full-struct.out

echo "compiling simple-full-struct-2.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  simple-full-struct-2.f95 -o simple-full-struct-2.out

echo "compiling simple-full-struct-implicit.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  simple-full-struct-implicit.f95 -o simple-full-struct-implicit.out

echo "compiling simple-full-struct-implicit-2.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  simple-full-struct-implicit-2.f95 -o simple-full-struct-implicit-2.out

echo "compiling pointer-target-map.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  pointer-target-map.f95 -o pointer-target-map.out

echo "compiling pointer-map.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  pointer-map.f95 -o pointer-map.out

echo "compiling nd-pointer-bounds-map-syntax.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nd-pointer-bounds-map-syntax.f95 -o nd-pointer-bounds-map-syntax.out

echo "compiling nd-allocatables-bounds-map-syntax.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nd-allocatables-bounds-map-syntax.f95 -o nd-allocatables-bounds-map-syntax.out

echo "compiling pointer-map-scopes.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  pointer-map-scopes.f95 -o pointer-map-scopes.out

echo "compiling pointer-scopes-enter-exit-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  pointer-scopes-enter-exit-map.f90 -o pointer-scopes-enter-exit-map.out

echo "compiling pointer-map-scopes-bounds.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  pointer-map-scopes-bounds.f95 -o pointer-map-scopes-bounds.out

echo "compiling pointer-array-section-1d-upperbound.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  pointer-array-section-1d-upperbound.f95 -o pointer-array-section-1d-upperbound.out

echo "compiling pointer-target-map-scopes.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  pointer-target-map-scopes.f95 -o pointer-target-map-scopes.out

echo "compiling pointer-target-scopes-enter-exit-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  pointer-target-scopes-enter-exit-map.f90 -o pointer-target-scopes-enter-exit-map.out

echo "compiling pointer-target-map-scopes-bounds.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  pointer-target-map-scopes-bounds.f95 -o pointer-target-map-scopes-bounds.out

echo "compiling pointer-target-array-section-1d-upperbound.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  pointer-target-array-section-1d-upperbound.f95 -o pointer-target-array-section-1d-upperbound.out

echo "compiling allocatable-map.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  allocatable-map.f95 -o allocatable-map.out 

echo "compiling nd-allocatables-target-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nd-allocatables-target-map.f90 -o nd-allocatables-target-map.out 

echo "compiling array-section-1d-upperbound.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  array-section-1d-upperbound.f95 -o array-section-1d-upperbound.out 

echo "compiling nd-bounds-map-syntax.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nd-bounds-map-syntax.f95 -o nd-bounds-map-syntax.out 

echo "compiling array-section-runtime-bounds.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  array-section-runtime-bounds.f95 -o array-section-runtime-bounds.out 

echo "compiling array-section-no-lower-bounds.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  array-section-no-lower-bounds.f95 -o array-section-no-lower-bounds.out 

echo "compiling nd-array-full-map.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nd-array-full-map.f95 -o nd-array-full-map.out 

echo "compiling assumed-size-array-vec-mul-allocatables.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  assumed-size-array-vec-mul-allocatables.f95 -o assumed-size-array-vec-mul-allocatables.out 

echo "compiling assumed-shape-array-vec-mul-allocatables.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  assumed-shape-array-vec-mul-allocatables.f95 -o assumed-shape-array-vec-mul-allocatables.out 

echo "compiling assumed-shape-array-vec-mul.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  assumed-shape-array-vec-mul.f95 -o assumed-shape-array-vec-mul.out 

echo "compiling assumed-size-array-vec-mul.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  assumed-size-array-vec-mul.f95 -o assumed-size-array-vec-mul.out 

echo "compiling target_enter_exit_milestone_3a.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  target_enter_exit_milestone_3a.f90 -o target_enter_exit_milestone_3a.out 

echo "compiling target_enter_exit_milestone_3b.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  target_enter_exit_milestone_3b.f90 -o target_enter_exit_milestone_3b.out 

echo "compiling allocatable-map-scopes.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  allocatable-map-scopes.f95 -o allocatable-map-scopes.out

echo "compiling allocatables-scopes-enter-exit-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  allocatables-scopes-enter-exit-map.f90 -o allocatable-scopes-enter-exit-map.out

echo "compiling allocatable-map-scopes-bounds.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  allocatable-map-scopes-bounds.f95 -o allocatable-map-scopes-bounds.out

echo "compiling allocatable-array-section-1d-upperbound.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  allocatable-array-section-1d-upperbound.f95 -o allocatable-array-section-1d-upperbound.out

echo "compiling target-alloca-from-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  target-alloca-from-map.f90 -o target-alloca-from-map.out 

echo "compiling enter-exit-break-test.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  enter-exit-break-test.f90 -o enter-exit-break-test.out 

echo "compiling enter-exit-break-test-2.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  enter-exit-break-test-2.f90 -o enter-exit-break-test-2.out 

echo "compiling target_enter_exit_milestone_3.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  target_enter_exit_milestone_3.f90 -o target_enter_exit_milestone_3.out

echo "compiling assumed-shape-array-vec-mul-allocatables-with-alloca-param.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  assumed-shape-array-vec-mul-allocatables-with-alloca-param.f95	 -o assumed-shape-array-vec-mul-allocatables-with-alloca-param.out

echo "compiling individual-dtype-member-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  individual-dtype-member-map.f90 -o individual-dtype-member-map.out

echo "compiling derived-type-individual-array-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  derived-type-individual-array-map.f90 -o individual-dtype-array-member-map.out

echo "compiling multiple-dt-explicit-member-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  multiple-dt-explicit-member-map.f90 -o multiple-dtype-member-map.out

echo "compiling derived-type-member-array-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  derived-type-member-array-map.f90 -o dt-member-array-map.out

echo "compiling full-dtype-map-with-contained-dtype.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  full-dtype-map-with-contained-dtype.f90 -o full-dtype-map-with-contained-dtype.out

echo "compiling dtype-member-array-map-2.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-member-array-map-2.f90 -o dt-member-array-map-2.out

echo "compiling derived-type-member-array-map-3.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  derived-type-member-array-map-3.f90 -o dt-member-array-map-3.out

echo "compiling double-derived-type-double-member-array-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  double-derived-type-double-member-array-map.f90 -o double-dtype-double-arr-map.out

echo "compiling double-derived-type-individual-member-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  double-derived-type-individual-member-map.f90 -o double-dtype-individual-scalar-map.out

echo "compiling derived-type-individual-array-map-with-bounds.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  derived-type-individual-array-map-with-bounds.f90 -o dtype-individual-array-map-with-bounds.out

echo "compiling double-dtype-mem-individual-arr-bounds-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  double-dtype-mem-individual-arr-bounds-map.f90 -o double-derived-type-individual-member-array-bounds-map.out

echo "compiling double-dtype-mixed-imp-exp-member-maps.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  double-dtype-mixed-imp-exp-member-maps.f90 -o dtype-exp-imp-member-map.out

echo "compiling double-dtype-mixed-imp-exp-member-maps-with-bounds.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  double-dtype-mixed-imp-exp-member-maps-with-bounds.f90 -o dtype-exp-imp-member-map-bounds.out

echo "compiling enter-exit-array-test.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  enter-exit-array-test.f90 -o enter-exit-array-test.out 

echo "compiling enter-exit-array-bounds-test.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  enter-exit-array-bounds-test.f90 -o enter-exit-array-bounds-test.out

echo "compiling enter-exit-scalar-test.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  enter-exit-scalar-test.f90 -o enter-exit-scalar-test.out

echo "compiling dtype-enter-exit.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-enter-exit.f90 -o dtype-enter-exit.out

echo "compiling dtype-enter-exit-update.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-enter-exit-update.f90 -o dtype-enter-exit-update.out

echo "compiling always-map-test.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  always-map-test.f90 -o always-map-test.out

echo "compiling allocatable-full-struct.f95"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  allocatable-full-struct.f95 -o allocatable-full-struct.out

echo "compiling declare-target-link-implicit.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  declare-target-link-implicit.f90 -o declare-target-link-implicit.out

echo "compiling implicit-allocatable-write-arrays.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  implicit-allocatable-write-arrays.f90 -o imp-alloca-write-arr.out 

echo "compiling double-dtype-nested-double-member-array-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  double-dtype-nested-double-member-array-map.f90 -o explicit-nested-array-map.out

echo "compiling large-nested-dtype-multi-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  large-nested-dtype-multi-map.f90 -o large-nested-dtype-multi-map.out

echo "compiling multi-large-nested-dtype-multi-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  multi-large-nested-dtype-multi-map.f90 -o multi-large-nested-dtype-multi-map.out

echo "compiling small-nested-dtype-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  small-nested-dtype-map.f90 -o small-nested-dtype-map.out

echo "compiling nested-dtype-single-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nested-dtype-single-map.f90 -o nested-dtype-single-map.out

echo "compiling multi-nested-dtype-single-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  multi-nested-dtype-single-map.f90 -o multi-nested-dtype-single-map.out

echo "compiling multi-nested-dtype-single-map-bounds.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  multi-nested-dtype-single-map-bounds.f90 -o multi-nested-dtype-single-map-bounds.out

echo "compiling multi-nested-dtype-multi-map-bounds.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  multi-nested-dtype-multi-map-bounds.f90 -o multi-nested-dtype-multi-map-bounds.out

echo "compiling nested-dtype-map-struct.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nested-dtype-map-struct.f90 -o nested-dtype-map-struct.out

echo "compiling nested-dtype-complex-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nested-dtype-complex-map.f90 -o nested-dtype-complex-map.out

echo "compiling dtype-array-allocatable.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-array-allocatable.f90 -o dtype-array-allocatable.out

echo "compiling dtype-allocatable-scalar-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-allocatable-scalar-map.f90 -o dtype-allocatable-scalar-map.out

echo "compiling dtype-allocatable-and-array-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-allocatable-and-array-map.f90 -o dtype-allocatable-and-array-map.out

echo "compiling dtype-allocatable-and-dtype-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-allocatable-and-dtype-map.f90 -o dtype-allocatable-and-dtype-map.out

echo "compiling dtype-allocatable-array-with-bounds-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-allocatable-array-with-bounds-map.f90 -o dtype-allocatable-array-with-bounds-map.out

echo "compiling nested-dtype-array-allocatable.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nested-dtype-array-allocatable.f90 -o nested-dtype-array-allocatable.out

echo "compiling nested-dtype-allocatable-and-array-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nested-dtype-allocatable-and-array-map.f90 -o nested-dtype-allocatable-and-array-map.out

echo "compiling nested-dtype-allocatable-array-with-bounds-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nested-dtype-allocatable-array-with-bounds-map.f90 -o nested-dtype-allocatable-array-with-bounds-map.out

echo "compiling nested-dtype-allocatable-and-dtype-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nested-dtype-allocatable-and-dtype-map.f90 -o nested-dtype-allocatable-and-dtype-map.out

echo "compiling allocatable-dtype-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  allocatable-dtype-map.f90 -o allocatable-dtype-map.out

echo "compiling dtype-array-of-dtype-member-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-array-of-dtype-member-map.f90 -o dtype-array-of-dtype-member-map.out

echo "compiling dtype-array-of-dtype-member-map-v2.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-array-of-dtype-member-map-v2.f90 -o dtype-array-of-dtype-member-map-v2.out

echo "compiling dtype-array-of-dtype-member-map-v3.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-array-of-dtype-member-map-v3.f90 -o dtype-array-of-dtype-member-map-v3.out

echo "compiling dtype-array-of-dtype-member-map-v4.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-array-of-dtype-member-map-v4.f90 -o dtype-array-of-dtype-member-map-v4.out

echo "compiling type_pointer_to_member.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype_pointer_to_member.f90 -o dtype_pointer_to_member.out

echo "compiling dtype-array-of-dtype-member-map-v5.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-array-of-dtype-member-map-v5.f90 -o dtype-array-of-dtype-member-map-v5.out

echo "compiling type-array-of-dtype-member-map-v6.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-array-of-dtype-member-map-v6.f90 -o dtype-array-of-dtype-member-map-v6.out

echo "compiling allocatable-dtype-nested-allocatable-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  allocatable-dtype-nested-allocatable-map.f90 -o allocatable-dtype-nested-allocatable-map.out

echo "compiling allocatable-dtype-and-nested-allocatable-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  allocatable-dtype-and-nested-allocatable-map.f90 -o allocatable-dtype-and-nested-allocatable-map.out

echo "compiling allocatable-dtype-and-nested-allocatable-map-v2.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  allocatable-dtype-and-nested-allocatable-map-v2.f90 -o allocatable-dtype-and-nested-allocatable-map-v2.out

echo "compiling nested-allocatable-dtype-allocatable-array-with-bounds-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  nested-allocatable-dtype-allocatable-array-with-bounds-map.f90 -o nested-allocatable-dtype-allocatable-array-with-bounds-map.out

echo "compiling michael-map-example.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  michael-map-example.f90 -o michael-map-example.out

echo "compiling target_map_common_block_1.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  target_map_common_block_1.f90 -o target_map_common_block_1.out

echo "compiling target_map_common_block_2.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  target_map_common_block_2.f90 -o target_map_common_block_2.out

echo "compiling target_map_common_block_3.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  target_map_common_block_3.f90 -o target_map_common_block_3.out

echo "compiling target_map_common_block_4.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  target_map_common_block_4.f90 -o target_map_common_block_4.out

echo "compiling target_map_common_block_5.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  target_map_common_block_5.f90 -o target_map_common_block_5.out

echo "compiling target_map_common_block_6.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  target_map_common_block_6.f90 -o target_map_common_block_6.out

echo "compiling target_map_common_block_7.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  target_map_common_block_7.f90 -o target_map_common_block_7.out

echo "compiling target_map_common_block_8.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  target_map_common_block_8.f90 -o target_map_common_block_8.out

echo "compiling allocatable-dtype-map-exp-member.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  allocatable-dtype-map-exp-member.f90 -o allocatable-dtype-map-exp-member.out

echo "compiling instruction-replace-test.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  instruction-replace-test.f90 -o instruction-replace-test.out

echo "compiling dtype-syntax-support.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-syntax-support.f90 -o dtype-syntax-support.out

echo "compiling dtype-syntax-support-2.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-syntax-support-2.f90 -o dtype-syntax-support-2.out

echo "compiling dtype-allocatable-array3d-with-bounds-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-allocatable-array3d-with-bounds-map.f90 -o dtype-allocatable-array3d-with-bounds-map.out

echo "compiling derived-type-map-from-alloca-issue.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  derived-type-map-from-alloca-issue.f90 -o derived-type-map-from-alloca-issue.out

echo "compiling dtype-class-dummy-full-exp-imp-map.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-class-dummy-full-exp-imp-map.f90 -o dtype-class-dummy-full-exp-imp-map.out

echo "compiling reproducer-for-471028.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  reproducer-for-471028.f90 -o reproducer-for-471028.out

echo "compiling enter-exit-in-different-scopes.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  enter-exit-in-different-scopes.f90 -o enter-exit-in-different-scopes.out

echo "compiling jean-example.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  jean-example.f90 -o jean-example.out

echo "compiling parent-member-overlap-test-1.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  parent-member-overlap-test-1.f90 -o parent-member-overlap-test-1.out

echo "compiling simple-declare-target-to-target.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  simple-declare-target-to-target.f90 -o simple-declare-target-to-target.out

echo "compiling declare-target-allocatable-array.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  declare-target-allocatable-array.f90 -o declare-target-allocatable-array.out

echo "compiling char-array-map-test-v1.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  char-array-map-test-v1.f90 -o char-array-map-test-v1.out

echo "compiling char-array-map-test-v2.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  char-array-map-test-v2.f90 -o char-array-map-test-v2.out

echo "compiling dtype-index-access-test.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  dtype-index-access-test.f90 -o dtype-index-access-test.out

echo "compiling neg-bounds-alloca-test.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  neg-bounds-alloca-test.f90 -o neg-bounds-alloca-test.out

echo "compiling swdev-561059.f90"

$AOMP/bin/flang    --offload-arch=gfx90a -fopenmp  swdev-561059.f90 -o swdev-516059.out

# these all depend on the fortran runtime compiled for offload

echo "compiling large-nested-allocatable-map-test.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  large-nested-allocatable-map-test.f90 -o large-nested-allocatable-map-test.out

echo "compiling multi-large-nested-dtype-multi-map-with-allocatables.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  multi-large-nested-dtype-multi-map-with-allocatables.f90 -o multi-large-nested-dtype-multi-map-with-allocatables.out

echo "compiling allocatable-dtype-array-map-v1.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  allocatable-dtype-array-map-v1.f90 -o allocatable-dtype-array-map-v1.out

echo "compiling large-mixed-nested-dtype-map.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  large-mixed-nested-dtype-map.f90 -o large-mixed-nested-dtype-map.out

echo "compiling large-mixed-nested-dtype-map-2.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  large-mixed-nested-dtype-map-2.f90 -o large-mixed-nested-dtype-map-2.out

echo "compiling large-allocatable-nested-dtype-multi-map.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  large-allocatable-nested-dtype-multi-map.f90 -o large-allocatable-nested-dtype-multi-map.out

echo "compiling large-allocatable-nested-dtype-double-map.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  large-allocatable-nested-dtype-double-map.f90 -o large-allocatable-nested-dtype-double-map.out

echo "compiling large-allocatable-nested-dtype-double-map-2.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  large-allocatable-nested-dtype-double-map-2.f90 -o large-allocatable-nested-dtype-double-map-2.out

echo "compiling allocatable-dtype-allocatable-nest-map.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  allocatable-dtype-allocatable-nest-map.f90 -o allocatable-dtype-allocatable-nest-map.out

echo "compiling large-nested-dtype-multi-map-with-allocatables.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  large-nested-dtype-multi-map-with-allocatables.f90 -o large-nested-dtype-multi-map-with-allocatables.out

echo "compiling nested-dtype-double-allocatable.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  nested-dtype-double-allocatable.f90  -o nested-dtype-double-allocatable.out

echo "compiling nested-dtype-scalar-allocatable.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  nested-dtype-scalar-allocatable.f90 -o nested-dtype-scalar-allocatable.out

echo "compiling dtype-scalar-allocatable.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  dtype-scalar-allocatable.f90 -o dtype-scalar-allocatable.out

echo "compiling dtype-double-allocatable.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  dtype-double-allocatable.f90  -o dtype-double-allocatable.out

echo "compiling implicit-allocatable-write.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  implicit-allocatable-write.f90 -o imp-alloca-write.out

echo "compiling single-value-alloca.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  single-value-alloca.f90 -o single-value-alloca.out

echo "compiling implicit-single-value-alloca.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  implicit-single-value-alloca.f90 -o implicit-single-value-alloca.out

echo "compiling nested-dtype-allocatable-scalar-map.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  nested-dtype-allocatable-scalar-map.f90 -o nested-dtype-allocatable-scalar-map.out

echo "compiling single-value-alloca-loop.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  single-value-alloca-loop.f90 -o single-value-alloca-loop.out

echo "compiling scalar-allocatable-block-map.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp  scalar-allocatable-block-map.f90 -o scalar-allocatable-block-map.out

echo "compiling UMT-reproducer.f90"

$AOMP/bin/flang --offload-arch=gfx90a -fopenmp  UMT-reproducer.f90 -o UMT-reproducer.out

echo "compiling test_map_types_omp60.F90"

$AOMP/bin/flang -fopenmp -fopenmp-version=60 --offload-arch=gfx90a test_map_types_omp60.F90 -o test_map_types_omp60.out

# USM tests that need HSA_XNACK set

echo "compiling declare-target-enter-usm.f90"

$AOMP/bin/flang -L$OFFLOAD_RUNTIME_DIR --offload-arch=gfx90a -fopenmp-force-usm -fopenmp  declare-target-enter-usm.f90 -o declare-target-enter-usm.out

echo "compiling UMT-reproducer-5.f90"

$AOMP/bin/flang --offload-arch=gfx90a -fopenmp  UMT-reproducer-5.f90 -o UMT-reproducer-5.out

echo "compiling UMT-reproducer-6.f90"

$AOMP/bin/flang --offload-arch=gfx90a -fopenmp  UMT-reproducer-6.f90 -o UMT-reproducer-6.out

echo "compiling reproducer-3-47779-v2.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  reproducer-3-47779-v2.f90 -o reproducer-3-47779-v2.out

echo "compiling reproducer-SWDEV-471201.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  reproducer-SWDEV-471201.f90 -o reproducer-SWDEV-471201.out

echo "compiling reproducer-476419.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  reproducer-476419.f90 -o reproducer-476419.out

echo "compiling reproducer-SWDEV-471201-v2.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  reproducer-SWDEV-471201-v2.f90 -o reproducer-SWDEV-471201-v2.out

echo "compiling reproducer-SWDEV-471201-v3.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  reproducer-SWDEV-471201-v3.f90 -o reproducer-SWDEV-471201-v3.out

echo "compiling pot3d-reproducer.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  pot3d-reproducer.f90 -o pot3d-reproducer.out

echo "compiling seismic-reproducer.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  seismic-reproducer.f90 -o seismic-reproducer.out

echo "compiling simplified-reproducer-497977.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  simplified-reproducer-497977.f90 -o simplified-reproducer-497977.out

echo "compiling reproducer-497977.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  reproducer-497977.f90 -o reproducer-497977.out

echo "compiling simplified-SWDEV-498999-reproducer.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  simplified-SWDEV-498999-reproducer.f90 -o simplified-SWDEV-498999-reproducer.out

echo "compiling simplified-SWDEV-498999-reproducer-v2.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  simplified-SWDEV-498999-reproducer-v2.f90 -o simplified-SWDEV-498999-reproducer-v2.out

echo "compiling SWDEV-564425-simple.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp SWDEV-564425-simple.f90 -o SWDEV-564425-simple.out

echo "compiling SWDEV-564425.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp SWDEV-564425.f90 -o SWDEV-564425.out

echo "compiling SWDEV-564425-v2.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp SWDEV-564425-v2.f90 -o SWDEV-564425-v2.out

echo "compiling SWDEV-564425-v3.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp SWDEV-564425-v3.f90 -o SWDEV-564425-v3.out

echo "compiling ref_ptr_ptee_test_1.f90"

$AOMP/bin/flang -fopenmp-version=60 -fopenmp --offload-arch=gfx90a ref_ptr_ptee_test_1.f90 -o ref_ptr_ptee_test_1.out

echo "compiling ref_ptr_ptee_test_2.f90"

$AOMP/bin/flang -fopenmp-version=60 -fopenmp --offload-arch=gfx90a ref_ptr_ptee_test_2.f90 -o ref_ptr_ptee_test_2.out

echo "compiling map_alloc_ref_ptr_ptee.f90"

$AOMP/bin/flang -fno-defer-desc-map -fopenmp-version=60 -fopenmp --offload-arch=gfx90a map_alloc_ref_ptr_ptee.f90 -o map_alloc_ref_ptr_ptee.out

echo "compiling map_static_ref_ptr_ptee.f90"

$AOMP/bin/flang -fno-defer-desc-map -fopenmp-version=60 -fopenmp --offload-arch=gfx90a map_static_ref_ptr_ptee.f90 -o map_static_ref_ptr_ptee.out

echo "compiling ref_ptr_ptee_struct_map-v1.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp ref_ptr_ptee_struct_map-v1.f90 -o ref_ptr_ptee_struct_map-v1.out

echo "compiling ref_ptr_ptee_struct_map-v2.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp ref_ptr_ptee_struct_map-v2.f90 -o ref_ptr_ptee_struct_map-v2.out

echo "compiling ref_ptr_ptee_alloca_struct_map-v1.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp ref_ptr_ptee_alloca_struct_map-v1.f90 -o ref_ptr_ptee_alloca_struct_map-v1.out

echo "compiling ref_ptr_ptee_alloca_struct_map-v2.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp ref_ptr_ptee_alloca_struct_map-v2.f90 -o ref_ptr_ptee_alloca_struct_map-v2.out

echo "compiling SWDEV-564425-v2-alloca-parent.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp SWDEV-564425-v2-alloca-parent.f90 -o SWDEV-564425-v2-alloca-parent.out

echo "compiling SWDEV-564425-v3-alloca-parent.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp SWDEV-564425-v3-alloca-parent.f90 -o SWDEV-564425-v3-alloca-parent.out

echo "compiling ref_ptr_ptee_alloca_struct_and-children_map-v1.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp ref_ptr_ptee_alloca_struct_and-children_map-v1.f90 -o ref_ptr_ptee_alloca_struct_and-children_map-v1.out

echo "compiling ref_ptr_ptee_alloca_struct_and-children_map-v2.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp ref_ptr_ptee_alloca_struct_and-children_map-v2.f90 -o ref_ptr_ptee_alloca_struct_and-children_map-v2.out

echo "compiling SWDEV-564425-v2-alloca-parent-and-children.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp SWDEV-564425-v2-alloca-parent-and-children.f90 -o SWDEV-564425-v2-alloca-parent-and-children.out

echo "compiling SWDEV-564425-v3-alloca-parent-and-children.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp SWDEV-564425-v3-alloca-parent-and-children.f90 -o SWDEV-564425-v3-alloca-parent-and-children.out

echo "compiling SWDEV-564425-v4-alloca-parent-and-children.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp SWDEV-564425-v4-alloca-parent-and-children.f90 -o SWDEV-564425-v4-alloca-parent-and-children.out

echo "compiling SWDEV-564425-v5-alloca-parent-and-children.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp SWDEV-564425-v5-alloca-parent-and-children.f90 -o SWDEV-564425-v5-alloca-parent-and-children.out

echo "compiling SWDEV-564425-v4-alloca-parent.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp SWDEV-564425-v4-alloca-parent.f90 -o SWDEV-564425-v4-alloca-parent.out

echo "compiling SWDEV-564425-v5-alloca-parent.f90"

$AOMP/bin/flang -fopenmp-version=60 --offload-arch=gfx90a -fopenmp SWDEV-564425-v5-alloca-parent.f90 -o SWDEV-564425-v5-alloca-parent.out

echo "compiling attach_always.f90"

$AOMP/bin/flang -fopenmp-version=61 -fopenmp --offload-arch=gfx90a attach_always.f90 -o attach_always.out

echo "compiling attach_never.f90"

$AOMP/bin/flang -fopenmp-version=61 -fopenmp --offload-arch=gfx90a attach_never.f90 -o attach_never.out

echo "compiling reproducer-SWDEV-483255.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  reproducer-SWDEV-483255.f90 -o reproducer-SWDEV-483255.out

echo "compiling reproducer-SWDEV-502486.f90"

$AOMP/bin/flang  --offload-arch=gfx90a -fopenmp  reproducer-SWDEV-502486.f90 -o reproducer-SWDEV-502486.out

echo "compiling SWDEV-579431.f90"

$AOMP/bin/flang -fopenmp -fopenmp-version=60 --offload-arch=gfx90a SWDEV-579431.f90 -o SWDEV-579431.out

echo "basic exp map"

./basic-exp-map.out

echo "basic example"

./basic-example.out

echo "basic example v2"

./basic-example-v2.out

echo "even basicer example"

./even-basicer-example.out

echo "main all"

./main-all.out

echo "main"

./main.out

echo "no write implicit cap"

./no-write-implicit-cap.out

echo "milestone 1 map syntax (1D range)"

./milestone-1-map-syntax.out

echo "exact milestone 1 map syntax (1D range)"

./milestone-1-map-exact-syntax.out

echo "double target call with same declare target map"

./decltar-double-target-call.out

echo "double target call with same basic map"

./basic-double-target-call.out

echo "Use of explicit constant index in target array"

./constant-index-in-target.out

echo "Use of from to to map arrays and transfer values"

./from-to.out 

echo "Use of complex value map from host to device"

./complex.out

echo "Complex array map from host to device"

./complex-array.out

echo "Constant array access and addition"

./constant-array-access.out

echo "Passing of basic struct in full"

./simple-full-struct.out

echo "Passing of basic struct in full, with individual member assignment"

./simple-full-struct-2.out

echo "Passing of basic implicit struct in full and write to other explicit struct"

./simple-full-struct-implicit.out

echo "Passing of basic implicit struct in full and write to other explicit struct, with individual member assignment"

./simple-full-struct-implicit-2.out

echo "Passing Fort pointer to Fort target and doing a simple loop assign"

./pointer-target-map.out 

echo "Passing Fort pointer and doing a simple loop assign"

./pointer-map.out 

echo "Passing Fort allocatable and doing a simple loop assign"

./allocatable-map.out 

echo "Check array section with upper bound (write off end of array on device)"

./array-section-1d-upperbound.out 

echo "N-D Bounds (3 dimensions) map syntax"

./nd-bounds-map-syntax.out

echo "1-D array runtime bounds"

./array-section-runtime-bounds.out

echo "1-D array section no LB specified with UB"

./array-section-no-lower-bounds.out

echo "3-D array full mapping, no bounds"

./nd-array-full-map.out

echo "1-D vec multiply in function with dummy assumed shape array arguments"

./assumed-shape-array-vec-mul.out

echo "1-D vec multiply in function with dummy assumed size array arguments"

./assumed-size-array-vec-mul.out

echo "1-D vec multiply in function with dummy assumed size array params with allocatable args"

./assumed-size-array-vec-mul-allocatables.out

echo "1-D vec multiply in function with dummy assumed shape array params with allocatable args"

./assumed-shape-array-vec-mul-allocatables.out

echo "Enter + Exit map used in conjunction with target to map an array and assign values"

./target_enter_exit_milestone_3a.out

echo "Enter + Exit map used in conjunction with target to map an allocatable array and assign values"

./target_enter_exit_milestone_3b.out

echo "Allocatable with Target utilising different variable scopes"

./allocatable-map-scopes.out

echo "Pointer with Target Region utilising different variable scopes"

./pointer-map-scopes.out

echo "Pointer+Target with Target Region utilising different variable scopes"

./pointer-target-map-scopes.out

echo "Allocatable with Target Region and Enter+Exit utilising different variable scopes"

./allocatable-scopes-enter-exit-map.out

echo "Pointer with Target Region and Enter+Exit utilising different variable scopes"

./pointer-scopes-enter-exit-map.out

echo "Pointer+Target with Target Region and Enter+Exit utilising different variable scopes"

./pointer-target-scopes-enter-exit-map.out

echo "3-D full Allocatable map with Target"

./nd-allocatables-target-map.out

echo "Regular Map Common Block and Utilise OMP builtin"

./target_map_common_block_1.out

echo "Regular Map Common Block and Assign Tmp"

./target_map_common_block_2.out

echo "1-D Allocatable Array sectioning "

./allocatable-array-section-1d-upperbound.out

echo "1-D Pointer Array sectioning "

./pointer-array-section-1d-upperbound.out

echo "1-D Pointer+Target Array sectioning "

./pointer-target-array-section-1d-upperbound.out

echo "1-D Allocatable Array with Array Sectioning in different scopes"

./allocatable-map-scopes-bounds.out

echo "1-D pointer Array with Array Sectioning in different scopes"

./pointer-map-scopes-bounds.out

echo "1-D pointer+target Array with Array Sectioning in different scopes"

./pointer-target-map-scopes-bounds.out

echo "3-D pointer map with bounds for target"

./nd-pointer-bounds-map-syntax.out

echo "3-D allocatables map with bounds for target"

./nd-allocatables-bounds-map-syntax.out

echo "Non-array integer allocatable map for target"

./single-value-alloca.out

echo "Non-array integer allocatable map for target with looped assignment/increment"

./single-value-alloca-loop.out

echo "Target Parallel do, teams distrbute, target and enter + exit"

./target_enter_exit_milestone_3.out 

echo "Target allocatable map with from "

./target-alloca-from-map.out

echo "Target allocatable map with enter+exit binding, remap of same variable and function invocation "

./enter-exit-break-test.out 

echo "Target allocatable map with enter+exit binding and re-map of same variable"

./enter-exit-break-test-2.out 

echo "Target in function with allocatable in parameter allocated then assigned to inside of function utilising target"

./assumed-shape-array-vec-mul-allocatables-with-alloca-param.out

echo "Mapping a single member of a derived type explicitly for a Target region"

./individual-dtype-member-map.out

echo "Mapping multiple members of a derived type explicitly for a Target region"

./multiple-dtype-member-map.out

echo "Mapping a single array member of a derived type explicitly for a Target region"

./individual-dtype-array-member-map.out

echo "Mapping two array members of a single derived type explicitly for a Target region"

./dt-member-array-map.out

echo "Mapping two 3-D array members of a single derived type explicitly for a Target region, with bounds defined"

./dt-member-array-map-2.out

echo "Mapping two 1-D array members of a single derived type explicitly for a Target region, with bounds defined"

./dt-member-array-map-3.out

echo "Mapping two 1-D array members of two derived types explicitly for a Target region, with bounds defined"

./double-dtype-double-arr-map.out

echo "Mapping a single scalar member from two derived types explicitly for a Target region"

./double-dtype-individual-scalar-map.out

echo "Mapping an individual array from a derived type with bounds to a Target region"

./dtype-individual-array-map-with-bounds.out

echo "Double derived type member map with one using explicit and other implicit"

./dtype-exp-imp-member-map.out

echo "Double derived type member array map with one using explicit bounds and other an implicit map"

./dtype-exp-imp-member-map-bounds.out

echo "Tests mapping of a regular 1-D array via enter and exit"

./enter-exit-array-test.out

echo "Tests mapping of a regular 1-D array with bounds via enter and exit"

./enter-exit-array-bounds-test.out

echo "Tests mapping of a regular scalar via enter and exit"

./enter-exit-scalar-test.out

echo "Tests derived type individual explicit member map for enter/exit"

./dtype-enter-exit.out

echo "Tests derived type individual explicit member map for enter/exit/update"

./dtype-enter-exit-update.out

echo "Test allocation of an array and then an always clause in combination with a tofrom map to get results"

./always-map-test.out

echo "Test two allocatable derived types being mapped with variables assigned"

./allocatable-full-struct.out

echo "Map double nested derived types with a scalar array by mapping the full top level derived type" 

./full-dtype-map-with-contained-dtype.out

echo "Implicit capture of declare target link (currently broken, Anchu should make a small PR for it)"

./declare-target-link-implicit.out

echo "Check if an implicit capture of an array allocatables is writeable"

./imp-alloca-write-arr.out

echo "Check if an implicit capture of an scalar allocatables is writeable"

./imp-alloca-write-arr.out

echo "Double dtype double nested array member mapping"

./explicit-nested-array-map.out

echo "Large nested derived type with multiple maps across all levels"

./large-nested-dtype-multi-map.out

echo "Large nested derived type with multiple maps across all levels including allocatable types"

./large-nested-dtype-multi-map-with-allocatables.out

echo "Double large nested derived type with multiple maps across all levels"

./multi-large-nested-dtype-multi-map.out

echo "Small nested derived type with mutiple maps across all levels"

./small-nested-dtype-map.out

echo "Small nested derived type with single nested map of array"

./nested-dtype-single-map.out

echo "Multi nested derived type with single nested map of array"

./multi-nested-dtype-single-map.out

echo "Multi nested derived type with single nested map of array with bounds"

./multi-nested-dtype-single-map-bounds.out

echo "Multi nested derived type with multi nested map of array with bounds"

./multi-nested-dtype-multi-map-bounds.out

echo "Nested derived type with a full nested struct map and other member maps"

./nested-dtype-map-struct.out

echo "Explicit member map of complex in a nested and non-nested dtype"

./nested-dtype-complex-map.out

echo "Explicit member map of scalar allocatable in a dtype"

./dtype-scalar-allocatable.out

echo "Explicit member map of scalar allocatable in a nested dtype"

./nested-dtype-scalar-allocatable.out

echo "Explicit member map of array allocatable in a dtype"

./dtype-array-allocatable.out

echo "Explicit member map of array allocatable in a nested dtype"

./nested-dtype-array-allocatable.out

echo "Explicit member map of array and scalar allocatable in a dtype at same time"

./dtype-double-allocatable.out

echo "Explicit member map of array and scalar allocatable in a nested dtype at same time"

./nested-dtype-double-allocatable.out

echo "Explicit member map of array allocatable and a regular scalar in a dtype at same time"

./dtype-allocatable-scalar-map.out

echo "Explicit member map of array allocatable and a regular scalar in a nested dtype at same time"

./nested-dtype-allocatable-scalar-map.out

echo "Explicit member map of array allocatable and a regular array in a dtype at same time"

./dtype-allocatable-and-array-map.out

echo "Explicit member map of array allocatable and a regular array in a nested dtype at same time"

./nested-dtype-allocatable-and-array-map.out

echo "Explicit member map of array allocatable and a dtype in a dtype at same time"

./dtype-allocatable-and-dtype-map.out

echo "Explicit member map of array allocatable and a dtype in a nested dtype at same time"

./nested-dtype-allocatable-and-dtype-map.out

echo "Explicit member map of array allocatable with bounds in a dtype"

./dtype-allocatable-array-with-bounds-map.out

echo "Explicit member map of array allocatable with bounds in a nested dtype"

./nested-dtype-allocatable-array-with-bounds-map.out

echo "Explicit map of allocatable dtype"

./allocatable-dtype-map.out

echo "Explicit member map of array of an allocatable dtype"

./allocatable-dtype-nested-allocatable-map.out

echo "Explicit member map of an allocatable dtype + allocatable array contained within"

./allocatable-dtype-and-nested-allocatable-map.out

echo "Explicit member map of an allocatable dtype + allocatable array contained within (arguments flipped)"

./allocatable-dtype-and-nested-allocatable-map-v2.out

echo "Allocatable derived type single explicit member map"

./allocatable-dtype-map-exp-member.out

echo "Derived type member map of an array of derived types"

./dtype-array-of-dtype-member-map.out

echo "Derived type member map of an array of derived types inside of allocatable dtype"

./dtype-array-of-dtype-member-map-v2.out

echo "Derived type member map of an allocatable array of derived types"

./dtype-array-of-dtype-member-map-v3.out

echo "Derived type member map of an allocatable array of derived types inside of allocatable dtype"

./dtype-array-of-dtype-member-map-v4.out

echo "Derived type member map with bounds of an allocatable array of derived types inside of allocatable dtype"

./dtype-array-of-dtype-member-map-v5.out

echo "Derived type member map with bounds of an array of derived types"

./dtype-array-of-dtype-member-map-v6.out

echo "Mapping of a single allocatable component of a nested allocatable derived type"

./allocatable-dtype-allocatable-nest-map.out

echo "Mapping two allocatable components of an nested allocatable derived type "

./large-allocatable-nested-dtype-double-map.out

echo "Mapping two allocatable components of an nested allocatable derived type, with multiple nested allocatable dtypes"

./large-allocatable-nested-dtype-double-map-2.out

echo "allocatable dtype with multiple maps of nested allocatable components"

./large-allocatable-nested-dtype-multi-map.out

echo "multiple allocatable dtype with multiple maps of nested allocatable components"

./multi-large-nested-dtype-multi-map-with-allocatables.out

echo "Map an dtype with allocatable array to device using enter/exit and then use a pointer to the array implicitly captured to write the data"

export LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS=1

./dtype_pointer_to_member.out

unset LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS

echo "Map an allocatable array nested in allocatable dtype with bounds"

./nested-allocatable-dtype-allocatable-array-with-bounds-map.out

echo "Map an allocatable dtype with a mix of allocatable and non-allocatable members nested in other dtypes"

./large-mixed-nested-dtype-map.out

echo "Map a dtype with a mix of allocatable and non-allocatable members nested in other dtypes"

./large-mixed-nested-dtype-map-2.out

echo "Example from Michael, arrays specified with dimensions contained in derived types mapped to a target and then iterated over"

./michael-map-example.out

echo "Map of full common block over multiple subroutines"

./target_map_common_block_3.out

echo "Map of first variable of common block over multiple subroutines"

./target_map_common_block_4.out

echo "Map of second variable of common block over multiple subroutines"

./target_map_common_block_5.out

echo "Map of all variables of a common block over multiple subroutines"

./target_map_common_block_6.out

echo "A mix of implicit and explicit mapping of common block members alongside a full common block map across subroutines"

./target_map_common_block_7.out

echo "Use of declare target link on a common block alongside a mix of implicit and explicit mapping of common block members alongside a full common block map across subroutines"

./target_map_common_block_8.out

echo "Test new instruction replacement removal does not break anything when used across multiple kernels"

./instruction-replace-test.out

echo "Test a series of derived type member mapping syntax with a non-allocatable derived type"

./dtype-syntax-support.out

echo "Test a series of derived type member mapping syntax with an allocatable derived type"

./dtype-syntax-support-2.out

echo "A map of multiple allocatable components from two seperate (but the same type) allocatable derived types"

./large-nested-allocatable-map-test.out

echo "A map of an allocatable N-D array within a derived type with specified bounds"

./dtype-allocatable-array3d-with-bounds-map.out

echo "Tests that 'from' mappings of allocatables do not segfault or memory access error on device"

./derived-type-map-from-alloca-issue.out

echo "Tests a block inside of a target with an allocatable scalar loop being assigned to, previously a buggy case now works"

./scalar-allocatable-block-map.out

echo "Tests implicit capture of a scalar allocatable is mapped as tofrom as specification requires "

./implicit-single-value-alloca.out

echo "Map a full derived type captured as a class dummy argument in two functions, for both impicit and explicit mapping"

./dtype-class-dummy-full-exp-imp-map.out

echo "Test that no regression of bug with enter/exit mapping of locally scoped arrays occurs, provided there is a matching delete for each enter"

./reproducer-for-471028.out

echo "Check that enter, exit and updates of an array spread across different calls and scopes works reasonably"

./enter-exit-in-different-scopes.out

echo "Test that Jean provided, which checks that we end up with the appropriate lower bound information when using raw input as map values"

./jean-example.out

echo "Reproducer v2 for clvrleaf error checking we do not overwrite data we don't map"

./reproducer-3-47779-v2.out

echo "Declare target enter/to mapping tests where mapped to target, allocatable/array/scalar tested"

./reproducer-SWDEV-471201.out

echo "Declare target enter/to mapping tests where mapped to target data and then updated, allocatable tested"

./reproducer-476419.out

echo "SWDEV-461201 reproducer, check if we have release error on update from with declare target to, resulting in segfault"

./reproducer-SWDEV-471201-v2.out

echo "SWDEV-471201 reproducer, make sure that we are correctly passing the relevant descriptor information across allowing appropriate assignment of values on device to the correct index"

./reproducer-SWDEV-471201-v3.out

echo "Regular Derived Type Member Map Test Checking "hole" punching where we map the parent and then some members seperately with different map types"

./parent-member-overlap-test-1.out

echo "This tests that calling an enter mapping twice before using it on target, doesn't break due to mapping the descriptor imlpicitly with (always, to)"

./pot3d-reproducer.out

echo "test obscure mapping condition that seismic reproducer triggers, which can clash with pot3d reproducer"

./seismic-reproducer.out

echo "test a simplified reproducer of SWDEV-497977"

./simplified-reproducer-497977.out

echo "test reproducer of SWDEV-497977"

./reproducer-497977.out

echo "Check that we at least don't explode when using THREADPRIVATE in the same program as a target region"

./simplified-SWDEV-498999-reproducer.out

echo "Check that we at least don't explode when using THREADPRIVATE in the same program as a target parallel teams distribute region"

./simplified-SWDEV-498999-reproducer-v2.out

echo "Check a simple usage of declare target to with target and update from"

./simple-declare-target-to-target.out

echo "Check a simple declare target to allocatable map to/from using target region and target data"

./declare-target-allocatable-array.out

echo "UMT Reproducer 1: Map nullary pointer to device and set target and modify target data"

./UMT-reproducer.out

echo "UMT Reproducer 5, make sure data is appropriately passed across over multiple enters of the same structure"

./UMT-reproducer-5.out

echo "UMT Reproducer 6 map back re-associate bug"

./UMT-reproducer-6.out

echo "Test mapping of allocatable char array in derived type"

./char-array-map-test-v1.out

echo "Test mapping of char array in derived type"

./char-array-map-test-v2.out

echo "Test intermediate index accessing works reasonably"

./dtype-index-access-test.out

echo "Test if negative bounds provided to allocatable derived type array causes problems"

./neg-bounds-alloca-test.out

echo "Test allocatable array offset for array slices"

./swdev-516059.out 100 100 1 1 30 100 31 60

./swdev-516059.out 100 101 1 1 30 101 31 60

./swdev-516059.out 50 101 2 1 30 100 31 60

./swdev-516059.out 300 201 20 1 30 101 31 60

./swdev-516059.out 300 201 20 10 50 141 41 90

echo "test a basic ref_ptr + ref_ptee example, case 1 (not a fool proof test as need to use smoke-test trace check examples)"

./ref_ptr_ptee_test_1.out

echo "test a basic ref_ptr + ref_ptee example, case 2 (not a fool proof test as need to use smoke-test trace check examples)" 

./ref_ptr_ptee_test_2.out

echo "test that checks ref_ptr works appropriately and allows us to circumvent the descriptor mapping issue by mapping just the data, example 1"

./map_alloc_ref_ptr_ptee.out

echo "test that checks ref_ptr works appropriately and allows us to circumvent the descriptor mapping issue by mapping just the data, example 2"

./map_static_ref_ptr_ptee.out

echo "Check ability to map components of derived type with ref_ptr/ptee v1"

./ref_ptr_ptee_struct_map-v1.out

echo "Check ability to map components of derived type with ref_ptr/ptee v2"

./ref_ptr_ptee_struct_map-v2.out

echo "Test we can enter map a derived types components in a block using ref_ptr/ptee and then execute a target region on the data"

./SWDEV-564425-v2.out

echo "Test we can enter map a derived types components in a block using ref_ptr/ptee and then execute a target region on the data v2"

./SWDEV-564425-v3.out

echo "attach always map test"

export LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS=0

./attach_always.out

unset LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS

echo "attach never map test"

LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS=0

./attach_never.out

unset LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS

echo "Test deallocation of a pointer array using ref_ptr/ptee"

./SWDEV-564425-simple.out

echo "Test deallocation of a pointer array inside of a derived type using ref_ptr/ptee"

./SWDEV-564425.out

echo "Test deallocation of multiple pointer members inside of a derived type using ref_ptr/ptee"

./SWDEV-564425-v2.out

echo "ref_ptr_ptee_alloca_struct_map-v1"

./ref_ptr_ptee_alloca_struct_map-v1.out

echo "ref_ptr_ptee_alloca_struct_map-v2"

./ref_ptr_ptee_alloca_struct_map-v2.out

echo "SWDEV-564425-v2-alloca-parent"

./SWDEV-564425-v2-alloca-parent.out

echo "SWDEV-564425-v3-alloca-parent"

./SWDEV-564425-v3-alloca-parent.out

echo "ref_ptr_ptee_alloca_struct_and-children_map-v1"

./ref_ptr_ptee_alloca_struct_and-children_map-v1.out

echo "ref_ptr_ptee_alloca_struct_and-children_map-v2"

./ref_ptr_ptee_alloca_struct_and-children_map-v2.out

echo "SWDEV-564425-v2-alloca-parent-and-children"

./SWDEV-564425-v2-alloca-parent-and-children.out

echo "SWDEV-564425-v3-alloca-parent-and-children"

./SWDEV-564425-v3-alloca-parent-and-children.out

echo "SWDEV-564425-v4-alloca-parent-and-children"

./SWDEV-564425-v4-alloca-parent-and-children.out

echo "SWDEV-564425-v5-alloca-parent-and-children"

./SWDEV-564425-v5-alloca-parent-and-children.out

echo "SWDEV-564425-v4-alloca-parent"

./SWDEV-564425-v4-alloca-parent.out

echo "SWDEV-564425-v5-alloca-parent"

export LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS=1

./SWDEV-564425-v5-alloca-parent.out

unset LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS

echo "Actually tests that derived type allcoatable storage is correctly released despite the name"

./test_map_types_omp60.out

# Tests that require XNACK/USM to pass

echo "test declare target enter/to usm works reasonably in simple cases"

export HSA_XNACK=1

./declare-target-enter-usm.out

unset HSA_XNACK

echo "SWDEV-579431 write-back issue caused by inter-mixing of implicit declare mapper"

./SWDEV-579431.out
