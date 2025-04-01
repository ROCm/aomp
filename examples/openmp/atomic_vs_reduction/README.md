# Atomic vs Reduction Example

This example (`atomic_vs_reduction.c`) demonstrates the relative speed of addition operations using:

1. **OpenMP Atomic with Hint**:
   - Uses `#pragma omp atomic hint(AMD_fast_fp_atomics)` to perform atomic addition with a hint for fast floating-point atomic operations.
   - This section is skipped if the compiler is not AMD's.

2. **OpenMP Atomic without Hint**:
   - Uses `#pragma omp atomic` for atomic addition without any hints.
   - Demonstrates the performance of atomic operations using a compare-and-swap (CAS) loop.

3. **OpenMP Reduction Operation**:
   - Uses `#pragma omp target teams distribute parallel for reduction(+:variable)` to perform a reduction operation.
   - Demonstrates the superior performance of reduction compared to atomic operations.

Each section calculates the sum of integers from `0` to `N-1` and compares the result with the expected value. The execution time for each method is also measured and displayed.

The example concludes that the reduction operation is superior.

## Building and Running the Example

### Prerequisites
Ensure that the following are set up:
1. The `LLVM_GPU_ARCH` environment variable is set to the target GPU architecture (e.g., `sm_30` for NVIDIA or `gfx90a` for AMD).
2. The `LLVM_INSTALL_DIR` environment variable points to the LLVM installation directory.

### Build Instructions
To build the example, run:
```bash
make
```

### Run Instructions
To execute the example, run:
```bash
make run
```

### Expected Output
The output will display the results of the three operations (atomic with hint, atomic without hint, and reduction) along with their execution times. A successful run will look like this:
```
Success atomic with hint (AMD_fast_fp_atomics) sum of <N> integers is: <SUM> in <TIME> secs  
Success atomic without hint (cas loop) sum of <N> integers is: <SUM> in <TIME> secs  
Success reduction sum of <N> integers is: <SUM> in <TIME> secs
```

If there are any mismatches in the calculated sums, the output will indicate a failure.
