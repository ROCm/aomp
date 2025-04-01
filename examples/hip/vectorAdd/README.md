vectorAdd
---------

This example shows how to add two vectors (along with the thread number) on the
target using HIP and the ROCm compiler.  The vectors are randomly generated and
the result is verified by computing the same operation on the host.

Example output:
```
  A: [1, 2, 8, 5, 8, 9, 2, 9, 0, 6]
  B: [6, 0, 1, 3, 3, 0, 9, 9, 7, 8]
Sum: [8, 4, 12, 12, 16, 15, 18, 26, 16, 24]
Success!
```

Some useful `make` commands for this example:
- `make run`: run the example
- `make clean`: clean the build files
- `make help`: see all available `make` targets and environment variables to
  customize your build. You can, for example, manually specify the LLVM and HIP
  installation directories:
  `LLVM_INSTALL_DIR=~/rocm/trunk HIPDIR=~/rocm/aomp make run`
