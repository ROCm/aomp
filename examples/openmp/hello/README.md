# OpenMP Hello World Examples

This directory contains five examples illustrating a simple "Hello World" program executed in different modes using OpenMP.

## Examples

1. **hello1**: Runs on the CPU.
2. **hello2**: Runs in parallel on the CPU, distributed across threads.
3. **hello3**: Runs on the GPU.
4. **hello4**: Runs in parallel on the GPU, distributed across threads.
5. **hello5**: Runs in parallel on the GPU, distributed across teams and threads.

## Prerequisites

- LLVM compiler with OpenMP support.
- A compatible GPU and the appropriate runtime environment.

## Build Instructions

To build all examples, run:

```bash
make
```

This will generate the following binaries:
- `hello1`
- `hello2`
- `hello3`
- `hello4`
- `hello5`

## Run Instructions

To execute all examples, use:

```bash
make run
```

This will run each binary and display the output for the respective example.

### Example Outputs

- **hello1**: Prints a message from the CPU and checks if it is running on the initial device.
- **hello2**: Prints messages from multiple threads running in parallel on the CPU.
- **hello3**: Prints a message from the GPU and checks if it is running on the initial device.
- **hello4**: Prints messages from multiple threads running in parallel on the GPU.
- **hello5**: Prints messages from multiple threads and teams running in parallel on the GPU.

## Cleanup

To clean up the generated files, run:

```bash
make clean
```

This will remove all binaries and intermediate files created during the build process.
