AOMP Developer README.md
========================

AOMP is a scripted build of LLVM and supporting software. It has support for OpenMP target offload on amdgcn GPUs.
This is the AOMP developer README stored at:
```
https://github.com/ROCm-Developer-Tools/aomp/blob/master/bin/README.md
```
The AOMP compiler supports OpenMP, clang-hip, clang-cuda, device OpenCL, and the offline kernel compilation tool called cloc.  It contains a recent version of the AMD Lightning compiler (llvm amdgcn backend) and the llvm nvptx backend.  Except for clang-cuda, this compiler works for both Nvidia and AMD Radeon GPUs.

This bin directory contains scripts to build and install AOMP from source.
```
clone_aomp.sh            -  A script to make sure the necessary repos are up to date.
                            See below for a list of these source repositories.
build_aomp.sh            -  This is the master build script. It runs all the
                            component build scripts in the correct order.
build_roct.sh            -  Builds the hsa thunk library.
build_rocr.sh            -  Builds the ROCm runtime.
build_project.sh         -  Builds llvm, lld, and clang components.
build_libdevice.sh       -  Builds the rocdl device bc libraries from rocm-device-libs.
build_comgr.sh           -  Builds the code object manager (needs rocm-device-libs).
build_rocminfo.sh        -  Builds the rocminfo utilities to support hip.
build_vdi.sh             -  Builds the hip interoperability layer ROCclr.
build_hipvdi.sh          -  Builds the hip host runtime using ROCclr.
build_extras.sh          -  Builds hostcall, libm, and utils all in aomp-extras repo.
build_openmp.sh          -  Builds the OpenMP libraries and device RTL.
build_pgmath.sh          -  Builds the pgmath support for flang.
build_flang.sh           -  Builds flang for aomp.
build_flang_runtime.sh   -  Builds the flang runtime for aomp.
build_ocl.sh             -  Builds OpenCL runtime.
build_rocdbgapi.sh       -  Builds ROCm Debugger API.
build_gdb.sh             -  Builds ROC gdb.
build_roctracer.sh       -  Builds ROC gdb. UNDER DEVELOPMENT

create_release_tarball.sh - This builds an important release artifact
                            containing all sources.
```

These scripts install into $HOME/rocm/aomp (or $AOMP if set).

## Repositories
<A NAME="Repositories">
The clone_aomp.sh script clones the necessary repositories and the correct
branches into subdirectories of $HOME/git/aomp13 (or $AOMP_REPOS if set)
The repositories needed by AOMP are shown in the following table.
The first column is the AOMP component that uses the repositories.

| Component | SUBDIRECTORY                           | REPOSITORY LINKS
| --------- | ------------                           | ----------------
| roct      | $HOME/git/aomp13/roct-thunk-interfaces | [roct-thunk-interfaces](https://github.com/radeonopencompute/roct-thunk-interface)
| rocr      | $HOME/git/aomp13/rocr-runtime          | [rocr-runtime](https://github.com/radeonopencompute/rocr-runtime)
| llvm-project | $HOME/git/aomp13/amd-llvm-project      | [llvm-project](https://github.com/ROCm-Developer-Tools/amd-llvm-project)
| extras    | $HOME/git/aomp13/aomp-extras           | [aomp-extras](https://github.com/ROCm-Developer-Tools/aomp-extras)
| vdi       | $HOME/git/aomp13/vdi                   | [vdi](https://github.com/ROCm-Developer-Tools/ROCclr)
| comgr     | $HOME/git/aomp13/rocm-compilersupport  | [comgr](https://github.com/radeonopencompute/rocm-compilersupport)
| hipvdi    | $HOME/git/aomp13/hip-on-vdi            | [hipvdi](https://github.com/ROCm-Developer-Tools/hip)
| ocl       | $HOME/git/aomp13/opencl-on-vdi         | [ocl](https://github.com/radeonopencompute/ROCm-OpenCL-Runtime)
| openmp    | $HOME/git/aomp13/llvm-project/openmp   | [llvm-project/openmp](https://github.com/ROCm-Developer-Tools/llvm-project)
| libdevice | $HOME/git/aomp13/rocm-device-libs      | [rocm-device-libs](https://github.com/radeonopencompute/rocm-device-libs)
| flang     | $HOME/git/aomp13/flang                 | [flang](https://github.com/ROCm-Developer-Tools/flang)
| rocminfo  | $HOME/git/aomp13/rocminfo              | [rocminfo](https://github.com/radeonopencompute/rocminfo)
| rocdbgapi | $HOME/git/aomp13/ROCdbgapi             | [rocdbgapi](https://github.com/ROCm-Developer-Tools/ROCdbgapi)
| rocgdb    | $HOME/git/aomp13/ROCgdb                | [rocgdb](https://github.com/ROCm-Developer-Tools/ROCgdb)
| roctrace  | $HOME/git/aomp13/roctracer             | [roctracer](https://github.com/ROCm-Developer-Tools/roctracer)
| rocprofiler | $HOME/git/aomp13/rocprofiler         | [rocprofiler](https://github.com/ROCm-Developer-Tools/rocprofiler)

## AOMP Environment Variables
The AOMP build scripts use many environment variables to control how AOMP is built.
The file [aomp_common_vars](aomp_common_vars) is sourced by all build scripts to provide consistent values
and default values for all environment variables.
There are generous comments in aomp_common_vars that provide information about the variables.
Developers should take some time to read comments in the [aomp_common_vars file](aomp_common_vars).

These are some important environment variables and their default values.
```
   AOMP                  $HOME/rocm/aomp
   CUDA                  /usr/local/cuda
   AOMP_REPOS            $HOME/git/aomp13
   AOMP_STANDALONE_BUILD 1
   AOMP_VERSION          13.0
   NVPTXGPUS             30,35,50,60,61,70
   GFXLIST               gfx700 gfx701 gfx801 gfx803 gfx900 gfx902 gfx906 gfx908
   BUILD_TYPE            Release
```
You can override any environment variable by setting values in your .bashrc or .bash_profile.
Here is a sample of commands you might want to put into your .bash_profile:
```
AOMP=/tmp/$USER/git/aomp
BUILD_TYPE=Debug
NVPTXGPUS=30,35,70
GFXLIST="gfx803 gfx906"
AOMP_VERSION="12.0"
export AOMP BUILD_TYPE NVPTXGPUS GFXLIST
```
## Quick Start to AOMP Development
To build and clone all components using the latest development sources, first clone aomp repo and checkout the aomp-dev branch as follows:

```
   mkdir $HOME/git/aomp13
   cd git/aomp13
   git clone https://github.com/ROCm-Developer-Tools/aomp.git aomp
   cd aomp
   git checkout aomp-dev
   git pull
   ./clone_aomp.sh
```
The first time you run ./clone_aomp.sh, it could take a long time to clone the repositories.
Subsequent calls will only pull the latest updates.

WARNING: The script clone_aomp.sh does not pull updates for this aomp repository. You must pull aomp repository manually.
So please run "git pull" frequently to stay current with the aomp development team.

```
cd $HOME/git/aomp13/aomp
git pull
./clone_aomp.sh
```

Building from source requires many prerequisites. Follow all the instructions [here](../docs/SOURCEINSTALL_PREREQUISITE.md)
to satisfy the many build-from-source prerequisites.
If you are on a system that another user has verified all the build-from-source requirements,
you still must install individual python components identified in section "2. User-installed Python Components".

After you have all the source repositories from running clone_aomp.sh and you have all the prerequisites installed,
run the master build script that steps through the build of all aomp components:
```
   ./build_aomp.sh
```
The default AOMP source build is a standalone build of all components needed for compilation and execution with the exception of the kfd Linux kernel module for AMD GPUs or the CUDA SDK for Nvidia GPUs.

## AOMP and the ROCM Compiler
Starting with ROCM 3.0 there is a version of AOMP distributed with ROCm with the package name aomp-amdgpu.
The ROCm aomp-amdgpu package installs into /opt/rocm/aomp with the command 'apt-get install aomp-amdgpu'.
Unlike the AOMP standalone build, ROCm aomp-amdgpu is integrated into ROCm.
It depends and uses existing ROCm components.  It was build with AOMP_STANDALONE_BUILD=0.

Eventually, the ROCm production compiler will have complete OpenMP support and the ROCm aomp-amdgpu package will be deprecated.
That ROCm production compiler has the package name llvm-amdgp and installs in /opt/rocm/llvm.

Starting with ROCM 4.0, AOMP will continue as a research and development compiler released into github with the package name aomp. 
This will often be used to validate code going into the ROCm production compiler including quick fixes for issues identified in
[AOMP issues] (https://github.com/ROCm-Developer-Tools/aomp/issues).
To ensure complete isolation from the ROCm installation and to make AOMP work in the absense of ROCm,
all necessary ROCm components are built from source.
Furthermore, through comprehensive use of RPATH by AOMP, all AOMP libraries and references to AOMP libraries will resolve the absolute location
within the AOMP installation.
AOMP never requires the use of LD_LIBRARY_PATH. It is reserved for user library development.
This strategy prevents any interference with a ROCm installation.
Libraries that may have been installed by a ROCm installation including roct and rocr, will not be used by AOMP.
Nor will the AOMP installation affect any ROCm component.

The only ROCm common component required by AOMP is the kernel kfd.


## Individual Component Builds
To build aomp, run the master build script build_aomp.sh or run these individual scripts in this order:

```
   # Start with these components if building AOMP standalone
   # AOMP_STANDALONE_BUILD==1 is the default.
   # If you turn this off, you must have a compatible ROCm installed.
   ./build_roct.sh
   ./build_roct.sh install
   ./build_rocr.sh
   ./build_rocr.sh install

   # These 6 core components are always built.
   # These are the only components built when AOMP_STANDALONE_BUILD=0.
   # The components pgmath, flang, and flang_runtime all use the flang repository.
   ./build_project.sh
   ./build_project.sh install
   ./build_extras.sh
   ./build_extras.sh install
   ./build_openmp.sh
   ./build_openmp.sh install
   ./build_pgmath.sh
   ./build_pgmath.sh install
   ./build_flang.sh
   ./build_flang.sh install
   ./build_flang_runtime.sh
   ./build_flang_runtime.sh install

   ./build_libdevice.sh
   ./build_libdevice.sh install
   ./build_comgr.sh
   ./build_comgr.sh install
   ./build_rocminfo.sh
   ./build_rocminfo.sh install

   # These only build on x86_64
   ./build_vdi.sh
   ./build_vdi.sh install
   ./build_hipvdi.sh
   ./build_hipvdi.sh install
   ./build_ocl.sh
   ./build_ocl.sh install

   # These two components build only when
   # AOMP_BUILD_STANDALONE==1 && AOMP_BUILD_DEBUG==1
   # which are the default values.
   ./build_rocdbgapi.sh
   ./build_rocdbgapi.sh install
   ./build_rocgdb.sh
   ./build_rocgdb.sh install
```

Once you have a successful development build, individual components can be incrementally rebuilt without rebuilding the entire system or the entire component. For example, if you change a file in the llvm-project repository. Run this command to incrementally build llvm, clang, and lld and update your installation.
```
   ./build_project.sh install
```
WARNING: When the build scripts are run with NO arguments (that is, you do not specify "install" or "nocmake"), the build scripts will rebuild the entire component by DELETING THE BUILD DIRECTORY before running cmake and make. This is called a fresh start.

## The AOMP Install Location
The build scripts will build from the source directories identified by the environment variable AOMP_REPOS.
The AOMP_REPOS default value is $HOME/git/aomp13.
The out-of-source build directory for each component is $AOMP_REPOS/build/\<component_name\>.

The install location is defined by the $AOMP environment variable. The value of AOMP MUST be reserved as a symbolic link.
That is, the physical installation will be in directory name formed by concatonating the version string to the value of $AOMP.
The "build_project.sh install" script will make a symbolic link from the physical directory to the symbolic directory $AOMP.
The default value for AOMP is $HOME/rocm/aomp.
For example, when building AOMP version 13.0-2 the install scripts will put all files and directories
in $HOME/rocm/aomp_13.0-2 and create a symbolic link as follows:

```
ln -sf ${AOMP}_13.0-2 ${AOMP}
```
All testing for AOMP uses the environment variable AOMP to locate the installation. This makes it easy to switch between versions of AOMP for testing by simply changing the environment variable AOMP. You do NOT need to change the symbolic link.
For example, if the aomp symbolic link currently points to aomp_13.0-2 and you want to test aomp_11.8-0, do this:

```
export AOMP=$HOME/rocm/aomp_11.8-0
```

The aomp package installs in /usr/lib/aomp_\<version_string\> and symlinks /usr/lib/aomp to the versioned directory. To test the installed package, set AOMP to /usr/lib/aomp or /usr/lib/aomp_\<version_string\>.

It is also possible to test the ROCm compiler with AOMP tests by setting AOMP as follows:

```
export AOMP=/opt/rocm/llvm
```

Many tests must determine the type of GPU available with the mygpu script.  The location of mygpu is expected in $AOMP/bin/mygpu. ROCm puts this utility in a different location.  But this script is only used if the environment variable AOMP_GPU is not set which is the default. So, to use AOMP tests with the ROCM compiler set the value of AOMP_GPU as follows:

```
export AOMP_GPU=`/opt/rocm/bin/mygpu`
```

## The AOMP developer patch subsystem

Of the many repos identified above, AOMP developers may only change amd-llvm-project, aomp, aomp-extras, and flang.  AOMP required changes to other non-AOMP components must be patched. The [patches/README.md](patches/README.md)  escribes the AOMP patch mechanism in detail.

## Building debuggable compiler

Two methods of building a debug (gdb) compiler
1) full build using export BUILD_TYPE=Debug when building llvm-project
   BUILD_TYPE=Debug ./build_aomp.sh select project

2) Selective build of files within llvm-project by using CMake option
  set_source_files_properties(ToolChains/CommonArgs.cpp PROPERTIES COMPILE_FLAGS "-O0 -g")

A full debug build is very expensive time wise, the link steps are much bigger memory footprints. I you have 64 GB of memory typically make -j16 will succeed.
Otherwise consider limiting the number of concurrent link steps. too many can exceed available system memory.

-DLLVM_PARALLEL_LINK_JOBS:2


A selective build using 'set_source_files...' will compile and link very quickly, and if combined with ./build_project.sh nocmake ; ./build_project.sh install
will be a more pleasant user experience.

