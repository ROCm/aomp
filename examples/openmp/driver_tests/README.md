# Driver Tests

This example demonstrates various compiler options and their outputs when using OpenMP target offloading. It showcases how to generate intermediate representations (IR), assembler files, and application binaries for both host and device.

## Prerequisites

- LLVM compiler with OpenMP support.
- A compatible GPU and the appropriate runtime environment.

## Build Instructions

To build the example and generate various outputs, run the following command:

```bash
make
```

This will generate:
1. Host LLVM IR (`driver_tests_host.ll`).
2. Host LLVM bitcode (`driver_tests_host.bc`).
3. Host assembler (`driver_tests_host.s`).
4. Device LLVM bitcode (`driver_tests_device.bc`).
5. Disassembled device LLVM IR (`driver_tests_device.ll`).
6. Application binary (`driver_tests`).

## Run Instructions

To execute the application binary, use the following command:

```bash
make run
```

This will run the binary and display information about the host and target devices, including:
- Initial and default devices.
- Number of devices.
- Threads and teams on the target device.

## Cleanup

To clean up the generated files, run:

```bash
make clean
```

This will remove all binaries and intermediate files created during the build process.