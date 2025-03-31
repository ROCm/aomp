# Declare Variant with Conditional Target Offload Example

This example demonstrates the following:

1. You must set `-fopenmp-version=50` for OpenMP 5.0 functionality.
2. How to use the `if` clause in pragmas for conditional target offload.
3. How to use function variants for different architectures.

## Example Description

The example implements the SAXPY (Single-Precision A·X Plus Y) operation using OpenMP. It demonstrates:

- Conditional offloading using the `if` clause in `#pragma omp target`.
- Architecture-specific function variants using `#pragma omp declare variant`.

The base `saxpy` function runs on the host, while architecture-specific variants (`amdgcn_saxpy` and `nvptx_saxpy`) are executed on AMD and NVIDIA GPUs, respectively.

## Build Instructions

To build the example, run the following command:

```bash
make
```

Ensure that the `LLVM_GPU_ARCH` environment variable is set to the appropriate GPU architecture (e.g., `gfx90a` for AMD GPUs or `sm_70` for NVIDIA GPUs).

## Execution Instructions

To execute the example, run:

```bash
make run
```

## Expected Output

The output will vary depending on the target architecture and the value of the `if` clause in the `#pragma omp target` directive. A typical output might look like:

```
Calling saxpy with high threshold for device execution
saxpy: Running on host. IsHost:1
Calling saxpy with low threshold for device execution
amdgcn_saxpy: Running on amdgcn device. IsHost:0
y[0],y[N-1]:     5   640
```
