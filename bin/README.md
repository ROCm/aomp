AOMP Developer README.md
========================

AOMP is a scripted build of LLVM and supporting software. It has support for OpenMP target offload on amdgcn GPUs.
This is the AOMP developer README stored at:
```
https://github.com/ROCm-Developer-Tools/aomp/blob/aomp-dev/bin/README.md
```
The AOMP compiler supports OpenMP, clang-hip, clang-cuda, device OpenCL, and the offline kernel compilation tool called cloc.  It contains a recent version of the AMD Lightning compiler (llvm amdgcn backend) and the llvm nvptx backend.  Except for clang-cuda, this compiler works for both Nvidia and AMD Radeon GPUs.


## Repositories
<A NAME="Repositories">
The clone_aomp.sh script clones the necessary github repositories and the correct
branches into subdirectories of $HOME/git/aomp15.0 (or $AOMP_REPOS if AOMP_REPOS is set).
The repositories and components needed by AOMP are shown in the following table.
The first column is the AOMP component name.  The build_aomp.sh script invokes
each component build script with the name build_\<component name\>.sh .


| COMPONENT | DEV BRANCH | DEFAULT DIRECTORY LOCATION           | REPOSITORY LINKS
| --------- | ---------- | --------------------------           | ----------------
| (aomp)    | aomp-dev   | $HOME/git/aomp15.0/aomp                | [aomp](https://github.com/ROCm-Developer-Tools/aomp) This repo!
| project   | aomp-dev   | $HOME/git/aomp15.0/llvm-project        | [llvm-project](https://github.com/ROCm-Developer-Tools/llvm-project)
| openmp    | aomp-dev   | $HOME/git/aomp15.0/llvm-project/openmp | [llvm-project/openmp](https://github.com/ROCm-Developer-Tools/llvm-project)
| extras    | aomp-dev   | $HOME/git/aomp15.0/aomp-extras         | [aomp-extras](https://github.com/ROCm-Developer-Tools/aomp-extras)
| pgmath    | aomp-dev   | $HOME/git/aomp15.0/flang/runtime/libpgmath | [flang](https://github.com/ROCm-Developer-Tools/flang)
| flang     | aomp-dev   | $HOME/git/aomp15.0/flang               | [flang](https://github.com/ROCm-Developer-Tools/flang)
| flang_runtime | aomp-dev | $HOME/git/aomp15.0/flang             | [flang](https://github.com/ROCm-Developer-Tools/flang)
|            |           |                                        |
| roct       |Latest ROCm| $HOME/git/aomp15.0/roct-thunk-interfaces | [roct-thunk-interfaces](https://github.com/radeonopencompute/roct-thunk-interface)
| rocr       |Latest ROCm| $HOME/git/aomp15.0/rocr-runtime        | [rocr-runtime](https://github.com/radeonopencompute/rocr-runtime)
| vdi        |Latest ROCm| $HOME/git/aomp15.0/vdi                 | [ROCclr](https://github.com/ROCm-Developer-Tools/ROCclr)
| comgr      |Latest ROCm| $HOME/git/aomp15.0/rocm-compilersupport| [comgr](https://github.com/radeonopencompute/rocm-compilersupport)
| hipvdi     |Latest ROCm| $HOME/git/aomp15.0/hip-on-vdi          | [hipvdi](https://github.com/ROCm-Developer-Tools/hip)
| ocl        |Latest ROCm| $HOME/git/aomp15.0/opencl-on-vdi       | [ocl](https://github.com/radeonopencompute/ROCm-OpenCL-Runtime)
| libdevice  |Latest ROCm| $HOME/git/aomp15.0/rocm-device-libs    | [rocm-device-libs](https://github.com/radeonopencompute/rocm-device-libs)
| rocminfo   |Latest ROCm| $HOME/git/aomp15.0/rocminfo            | [rocminfo](https://github.com/radeonopencompute/rocminfo)
| rocdbgapi  |Latest ROCm| $HOME/git/aomp15.0/ROCdbgapi           | [rocdbgapi](https://github.com/ROCm-Developer-Tools/ROCdbgapi)
| rocgdb     |Latest ROCm| $HOME/git/aomp15.0/ROCgdb              | [rocgdb](https://github.com/ROCm-Developer-Tools/ROCgdb)
| roctracer  |Latest ROCm| $HOME/git/aomp15.0/roctracer           | [roctracer](https://github.com/ROCm-Developer-Tools/roctracer)
| rocm-cmake |Latest ROCm| $HOME/git/aomp15.0/rocm-cmake          | [rocm-cmake](https://github.com/RadeonOpenCompute/rocm-cmake)
| rocm_smi_lib|Latest ROCm| $HOME/git/aomp15.0/rocm_smi_lib       | [rocm_smi_lib](https://github.com/RadeonOpenCompute/rocm_smi_lib)
| hwloc      |  v2.7     | $HOME/git/aomp15.0/hwloc               | [hwloc](https://github.com/open-mpi/hwloc)

If the component is a core aomp component, then the development branch name is aomp-dev.
The default branch name for other non-core components is set in the aomp_common_vars file.
It is typically the name of the latest ROCm release in the external source repository.

## AOMP Environment Variables
The AOMP build scripts use many environment variables to control how AOMP is built.
The file [aomp_common_vars](aomp_common_vars) is sourced by all build scripts to provide consistent values
and default values for all environment variables.
There are generous comments in aomp_common_vars that provide information about the variables.
Developers should take some time to read comments in the [aomp_common_vars file](aomp_common_vars).

These are some important environment variables and their default values.

| ENV VARIABLE | DEFAULT VALUE               | DESCRIPTION
| ------------ | -------------               | -----------
| AOMP                  | $HOME/rocm/aomp    | Where AOMP is installed and tested
| AOMP_REPOS            | $HOME/git/aomp15.0   | The base directory for all AOMP build repositories
| AOMP_STANDALONE_BUILD | 1                  | Build all components, do NOT use installed ROCm
| AOMP_VERSION          | 15.0               | Clang version.
| AOMP_VERSION_MOD      | 2                  | This implies the next release will be AOMP_15.0-2.
| AOMP_VERSION_STRING   | $AOMP_VERSION-$AOMP_VERSION_MOD |
| GFXLIST               | gfx700 gfx701 gfx801 gfx803     | List of AMDGPU gpus to build for
|                       | gfx900 gfx902 gfx906 gfx908     |
| NVPTXGPUS             | 30,35,50,60,61,70               | List of NVPTX gpus to build for
| CUDA                  | /usr/local/cuda                 | Where cuda is installed


You can override any environment variable by setting values in your .bashrc or .bash_profile.
Here is a sample of commands you might want to put into your .bash_profile:

```
AOMP=/tmp/$USER/rocm/aomp
AOMP_REPOS=/tmp/$USER/git/aomp15.0
NVPTXGPUS=30,35,70
GFXLIST="gfx803 gfx906"
AOMP_VERSION="15.0"
export AOMP BUILD_TYPE NVPTXGPUS GFXLIST
```
## Quick Start to AOMP Development
To build and clone all components using the latest development sources, first clone aomp repo and checkout the aomp-dev branch as follows:

```
   mkdir $HOME/git/aomp15.0
   cd git/aomp15.0
   git clone https://github.com/ROCm-Developer-Tools/aomp.git aomp
   cd aomp
   git checkout aomp-dev
   git pull
   cd bin
   ./clone_aomp.sh
```
The first time you run ./clone_aomp.sh, it could take a long time to clone the repositories.
Subsequent calls will only pull the latest updates.

WARNING: The script clone_aomp.sh does not pull updates for this aomp repository. You must pull aomp repository manually.
So please run "git pull" frequently to stay current with the aomp development team.

```
cd $HOME/git/aomp15.0/aomp
git pull
cd bin
./clone_aomp.sh
```
AOMP_EXTERNAL_MANIFEST=1 can be used to ensure clone_aomp.sh does not use an internal manifest. This is only useful of the internal ping succeeds, otherwise the external manifest us used by default.

Building from source requires many prerequisites. Follow all the instructions [here](../docs/SOURCEINSTALL_PREREQUISITE.md)
to satisfy the many build-from-source prerequisites.
If you are on a system that another user has verified all the build-from-source requirements,
you still must install individual python components identified in section "2. User-installed Python Components".

After you have all the source repositories from running clone_aomp.sh and you have all the prerequisites installed,
run the master build script that steps through the build of all aomp components:
```
   ./build_aomp.sh
```
There are options to select a subset of components to build and install.
1) Start with openmp and continue to the end of the component list.
```
  ./build_aomp.sh continue openmp
```
2) Select a subset of components to build.
```
  ./build_aomp.sh select openmp flang flang_runtime
```
The default AOMP source build is a standalone build of all components needed for compilation and execution with the exception of the kfd Linux kernel module for AMD GPUs or the CUDA SDK for Nvidia GPUs.

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

This bin directory ($HOME/git/aomp15.0/aomp/bin) contains many component scripts to build and install AOMP from source.
Here are some of the scripts contained in this direcotry
```
clone_aomp.sh            -  A script to make sure the necessary repos are up to date.
                            See list source repositories above.
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
build_rocm-cmake.sh      -  Builds rocm-cmake.
build_rocm_smi_lib.sh    -  Builds rocm_smi_lib.
build_prereq.sh          -  Builds prerequisites such as cmake, hwloc, rocmsmi.

create_release_tarball.sh - This builds an important release artifact
                            containing all sources.
```
To build aomp, run the master build script build_aomp.sh, or run the individual scripts in the order shown below.
The build scripts with no arguments will build the component in the build directory $HOME/git/aomp15.0/build/\<component name\>
(or $AOMP_REPOS/build/\<component name\> if AOMP_REPOS is set).
The component build scripts take a single positional argument with the value "install" or "nocmake".
The master build script build_aomp.sh will call the component build scripts in the following order and stop if
any fails occur.

```
   # AOMP_STANDALONE_BUILD==1 is the default.
   # If you turn this off, you must have a compatible ROCm installed.
   # AOMP_STANDALONE_BUILD==0 only installs openmp, extras, pgmath, flang, flang_runtime.

   ./build_prereq.sh
   ./build_roct.sh
   ./build_roct.sh install
   ./build_rocr.sh
   ./build_rocr.sh install
   ./build_project.sh
   ./build_project.sh install
   ./build_libdevice.sh
   ./build_libdevice.sh install
   ./build_openmp.sh
   ./build_openmp.sh install
   ./build_extras.sh
   ./build_extras.sh install
   ./build_comgr.sh
   ./build_comgr.sh install
   ./build_rocminfo.sh
   ./build_rocminfo.sh install
   ./build_rocm-cmake.sh
   ./build_rocm-cmake.sh install

   # The components pgmath, flang, and flang_runtime all use the flang repository.
   ./build_pgmath.sh
   ./build_pgmath.sh install
   ./build_flang.sh
   ./build_flang.sh install
   ./build_flang_runtime.sh
   ./build_flang_runtime.sh install

   # These components only build on x86_64
   ./build_hipamd.sh
   ./build_hipamd.sh install

   # These two components build only when
   # AOMP_BUILD_STANDALONE=1 && AOMP_BUILD_DEBUG=1
   # which are the default values.
   ./build_rocdbgapi.sh
   ./build_rocdbgapi.sh install
   ./build_rocgdb.sh
   ./build_rocgdb.sh install
   ./build_roctracer.sh
   ./build_roctracer.sh install
   ./build_rocprofiler.sh
   ./build_rocprofiler.sh install
```

Once you have a successful development build, individual components can be incrementally rebuilt without rebuilding the entire system or the entire component. For example, if you change a file in the llvm-project repository. Run this command to incrementally build llvm, clang, and lld and update your installation.
```
   ./build_project.sh install
```
WARNING: When the build scripts are run with NO arguments (that is, you do not specify "install" or "nocmake"), the build scripts will rebuild the entire component by DELETING THE BUILD DIRECTORY before running cmake and make. This is called a FRESH start.

## The AOMP Install Location
The build scripts will build from the source directories identified by the environment variable AOMP_REPOS.
The AOMP_REPOS default value is currently $HOME/git/aomp15.0.
The out-of-source build directory for each component is $AOMP_REPOS/build/\<component_name\>.

The install location is defined by the $AOMP environment variable. The value of AOMP MUST be reserved as a symbolic link.
That is, the physical installation will be in directory name formed by concatonating the version string to the value of $AOMP.
The "build_project.sh install" script will make a symbolic link from the physical directory to the symbolic directory $AOMP.
The default value for AOMP is $HOME/rocm/aomp.
For example, when building AOMP version 15.0-1 the install scripts will put all files and directories
in $HOME/rocm/aomp_15.0-1 and create a symbolic link as follows:

```
ln -sf ${AOMP}_15.0-1 ${AOMP}
```
All testing for AOMP uses the environment variable AOMP to locate the installation. This makes it easy to switch between versions of AOMP for testing by simply changing the environment variable AOMP. You do NOT need to change the symbolic link.
For example, if the aomp symbolic link currently points to aomp_15.0-1 and you want to test aomp_15.0-0, do this:

```
export AOMP=$HOME/rocm/aomp_15.0-0
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

Of the many repos identified above, AOMP developers may only change core aomp repositories (llvm-project, aomp, aomp-extras, and flang).  AOMP required changes to other non-core components must be patched. The [patches/README.md](patches/README.md)  describes the AOMP patch mechanism in detail.

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

