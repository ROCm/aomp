vmulsum
-------

This is an illustration of how to compile multiple source files with the ROCm
compiler.

The code performs vector addition (`vsum`) and multiplication (`vmul`) on the
host and on the target.  OpenMP is used for offloading to the target and
executing the corresponding vector operations in parallel and distributed over
thread teams.  In the end, the computed results of the host and the target are
compared. When they match, the program prints `Success` and terminates
successfully.

To compile and run:
`make run`

For help:
`make help`
