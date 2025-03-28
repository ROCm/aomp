# Demo Offload Types

This example demonstrates the use of OpenMP target offloading with various configurations. It showcases how to compile and execute binaries for different offload types, including host-only, GPU offload, and GPU offload with `xnack+`.

## Prerequisites

- LLVM compiler with OpenMP support.
- A compatible GPU and the appropriate runtime environment.

## Build Instructions

To build the example, run the following command:

```bash
make
```

This will generate the following binaries:
- `no-offload`: Binary without offloading.
- `host-offload`: Binary for host offloading (if supported).
- `gpu-offload`: Binary for GPU offloading.
- `demo_offload_types`: Binary for GPU offloading with `xnack+` (if applicable).

## Run Instructions

To execute the binaries, use the following command:

```bash
make run
```

This will run the binaries in various configurations, including:
- Default execution.
- Execution with `OMP_TARGET_OFFLOAD=DISABLED`.
- Execution with `OMP_TARGET_OFFLOAD=MANDATORY`.

### Example Output

The output will display information about the host and target devices, including:
- Initial and default devices.
- Number of devices.
- Threads and teams on the target device.

## Cleanup

To clean up the generated files, run:

```bash
make clean
```

This will remove all binaries and intermediate files created during the build process.