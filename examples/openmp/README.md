Examples to demonstrate use of OpenMP in c and C++
==================================================

The LLVM compiler supports several accelerated programming models for GPUs.
OpenMP is one of these models.
Examples in this directory demonstrate how to compile and execute c and C++ applications
that use OpenMP target offload.

If examples are in a read-only directory, the Makefile can be run from a writeable directory as follows:
```
mkdir /tmp/demo ; cd /tmp/demo
EXROOT=/opt/rocm/share/openmp-extras/examples  # The examples base directory.
make -f $EXROOT/openmp/reduction/Makefile run
```
To execute an example directly, recursively copy the entire examples directory
to a writeable directory. For example:
```
EXROOT=/opt/rocm/share/openmp-extras/examples # The examples base directory.
cp -rp $EXROOT /tmp/                          # Recursively copy examples to /tmp/examples
cd /tmp/examples/openmp/reduction             # cd to writeable example directory
make run                                      # Compile and execute the reduction example
```
There are many other make targets to show different ways to build the binary. Run ```make help``` to see all the possible demos as Makefile targets.

E.g. to run with some debug output set OFFLOAD_DEBUG variable:

```
env OFFLOAD_DEBUG=1 make
env OFFLOAD_DEBUG=1 make run
```
These are the c and C++ examples in the [openmp](.) examples category:
- [veccopy](veccopy)
- [vmulsum](vmulsum)
- [reduction](reduction)
- [driver_tests](driver_tests)
- [declare_variant_if](declare_variant_if)
- [show-offload-types](show-offload-types)
- [vmul_template](vmul_template)
