AOMP Developer README.md
========================

AOMP is a scripted build of LLVM and supporting software. It has support for OpenMP target offload on amdgcn GPUs.
This is the AOMP developer README stored at:
```
https://github.com/ROCm/aomp/blob/aomp-dev/bin/README.md
```
The AOMP compiler supports OpenMP, clang-hip, clang-cuda, device OpenCL, and the offline kernel compilation tool called cloc.  It contains a recent version of the AMD Lightning compiler (llvm amdgcn backend) and the llvm nvptx backend.  Except for clang-cuda, this compiler works for both Nvidia and AMD Radeon GPUs.


## Repositories

<A NAME="Repositories"></A>
The clone_aomp.sh script clones the necessary github repositories and the correct
branches into subdirectories of $HOME/git/aomp20.0 (or $AOMP_REPOS if AOMP_REPOS is set).
The repositories and components needed by AOMP are shown in the following table.
The first column is the AOMP component name.  The build_aomp.sh script invokes
each component build script with the name build_\<component name\>.sh .


| COMPONENT | DEV BRANCH | DEFAULT DIRECTORY LOCATION           | REPOSITORY LINKS
| --------- | ---------- | --------------------------           | ----------------
| (aomp)    | aomp-dev   | $HOME/git/aomp20.0/aomp                | [aomp](https://github.com/ROCm/aomp) This repo!
| project   | amd-staging | $HOME/git/aomp20.0/llvm-project      | [llvm-project](https://github.com/ROCm/llvm-project)
| openmp    | amd-staging | $HOME/git/aomp20.0/llvm-project/openmp | [llvm-project/openmp](https://github.com/ROCm/llvm-project)
| extras    | aomp-dev   | $HOME/git/aomp20.0/aomp-extras         | [aomp-extras](https://github.com/ROCm/aomp-extras)
| flang     | aomp-dev   | $HOME/git/aomp20.0/flang               | [flang](https://github.com/ROCm/flang)
| pgmath    | aomp-dev   | $HOME/git/aomp20.0/flang/runtime/libpgmath | [flang](https://github.com/ROCm/flang)
| flang-legacy | aomp-dev   | $HOME/git/aomp20.0/flang/flang-legacy/17.0-4 | [flang](https://github.com/ROCm/flang/flang-legacy/17.0-4/llvm-legacy)
| flang_runtime | aomp-dev | $HOME/git/aomp20.0/flang             | [flang](https://github.com/ROCm/flang)
||           |                                        |
| roct       |Latest ROCm| $HOME/git/aomp20.0/roct-thunk-interfaces | [roct-thunk-interfaces](https://github.com/ROCm/roct-thunk-interface)
| rocr       |Latest ROCm| $HOME/git/aomp20.0/rocr-runtime        | [rocr-runtime](https://github.com/ROCm/rocr-runtime)
| hip        |Latest ROCm| $HOME/git/aomp20.0/hip              | [hipamd](https://github.com/ROCm/hip)
|            |Latest ROCm| $HOME/git/aomp20.0/hipcc                 | [hip](https://github.com/ROCm/hipcc)
|            |Latest ROCm| $HOME/git/aomp20.0/clr              | [ROCclr](https://github.com/ROCm/clr)
|            |Latest ROCm| $HOME/git/aomp20.0/ROCm-OpenCL-Runtime | [ocl](https://github.com/ROCm/ROCm-OpenCL-Runtime)
| comgr      |Latest ROCm| $HOME/git/aomp20.0/rocm-compilersupport| [comgr](https://github.com/ROCm/rocm-compilersupport)
| rocminfo   |Latest ROCm| $HOME/git/aomp20.0/rocminfo            | [rocminfo](https://github.com/ROCm/rocminfo)
| rocdbgapi  |Latest ROCm| $HOME/git/aomp20.0/ROCdbgapi           | [rocdbgapi](https://github.com/ROCm/ROCdbgapi)
| rocgdb     |Latest ROCm| $HOME/git/aomp20.0/ROCgdb              | [rocgdb](https://github.com/ROCm/ROCgdb)
| roctracer  |Latest ROCm| $HOME/git/aomp20.0/roctracer           | [roctracer](https://github.com/ROCm/roctracer)

Notice that some components are built with different parts of the same repository.

The [llvm-project](https://github.com/ROCm/llvm-project) repository is a mirror of the gerrit-managed internal AMD repository.
The updates to the mirror are done at least daily.
If you are internal to AMD you will pick up the llvm-project repository directly from the gerrit
 [llvm-project](ssh://git.amd.com:29418/lightning/ec/llvm-project) repository.

In addition to the gerrit-managed llvm-project repository, the three repositories
[aomp](https://github.com/ROCm/aomp) ,
[aomp-extras](https://github.com/ROCm/aomp-extras) , and
[flang](https://github.com/ROCm/flang) make up the core aomp development repositories.
These three use the branch "aomp-dev" and are managed completely on external github.
The default branch name for other non-core components is typically the name of the latest ROCm release in the external source repository.

The clone_aomp.sh script, described in the "Quick Start" below, uses a manfest file from the
[manifests](https://github.com/ROCm/aomp/tree/aomp-dev/manifests)
directory to find and clone all repositories used by aomp.

## AOMP Environment Variables
The AOMP build scripts use many environment variables to control how AOMP is built.
The file [aomp_common_vars](aomp_common_vars) is sourced by all build scripts to provide consistent values
and default values for all environment variables.
There are generous comments in aomp_common_vars that provide information about the variables.
Developers should take some time to read comments in the [aomp_common_vars file](aomp_common_vars).

These are some important environment variables and their default values.

| ENV VARIABLE | DEFAULT VALUE               | DESCRIPTION
| ------------ | -------------               | -----------
| AOMP                  | $HOME/rocm/aomp    | Directory symbolic link to where AOMP is installed and tested
| AOMP_REPOS            | $HOME/git/aomp20.0   | The base directory for all AOMP build repositories
| AOMP_STANDALONE_BUILD | 1                  | Build all components, do NOT use installed ROCm
| AOMP_VERSION          | 20.0               | Clang version.
| AOMP_VERSION_MOD      | 0                  | This implies the next release will be AOMP_20.0-0.
| AOMP_VERSION_STRING   | $AOMP_VERSION-$AOMP_VERSION_MOD |
| AOMP_USE_NINJA        | 0                  | Use ninja instead of make to build certain components
| GFXLIST               | gfx700 gfx701 gfx801 gfx803     | List of AMDGPU gpus to build for
|                       | gfx900 gfx902 gfx906 gfx908     |
| NVPTXGPUS             | 30,35,50,60,61,70               | List of NVPTX gpus to build for
| CUDA                  | /usr/local/cuda                 | Where cuda is installed


### Overriding AOMP Enviornment Variables
You can override any environment variable by setting values in your .bashrc or .bash_profile.
We do not recommend overriding any AOMP_VERSION values as these defaults are updated by development
leaders in the [aomp_common_vars](aomp_common_vars) file when development moves to the next release.

Here is a sample of commands you might want to put into your .bashrc or .bash_profile:

```
AOMP=/work/$USER/rocm/aomp
AOMP_REPOS=/work/$USER/git/aomp20.0
NVPTXGPUS=30,35,70
GFXLIST="gfx803 gfx906"
AOMP_USE_NINJA=1
export AOMP AOMP_REPOS NVPTXGPUS GFXLIST AOMP_USE_NINJA
```
This sample demonstrates how to set AOMP_REPOS to store the repositories and build directory
on a filesystem that may be faster than your HOME directory.
We recommend that, if you override AOMP_REPOS as shown, set AOMP_REPOS to a directory name
that contains the current default value AOMP_VERSION as set in the beginning
of the [aomp_common_vars](aomp_common_vars) file.
This sample also shows how to tell the build scripts to use Ninja instead of make for faster build times.

### The "AOMP" Environment Variable
The above sample sets AOMP to the location of the AOMP installation link.
AOMP is the most important environment variable to set and understand.
This environment variable is used by BOTH the build scripts and most test scripts.
The build scripts REQUIRE that AOMP is either a symbolic link or initially does not exist.
The build scripts will install to a physical directory with AOMP_VERSION_STRING
appended to the value of AOMP. The build scripts will create the symbolic link from AOMP
to the physical installation directory.

Most test scripts use the environment variable AOMP to identify the compiler location.
For testing purposes, you will often change the value of AOMP when not testing your latest
AOMP build.  For example, to test the latest  packaged aomp release, set AOMP=/usr/lib/aomp.
To test against the ROCM installed compiler, set AOMP=/opt/rocm/lib/llvm. To test one of your
previous AOMP source builds, set AOMP to an older physical installation directory, such as
AOMP=$HOME/rocm/aomp_18.0-1. BE CAREFUL! If you set AOMP for testing purposes,
remember to set it back before rebuilding aomp from source.

## Quick Start to AOMP Development
To build and clone all components using the latest development sources, first clone aomp repo and checkout the aomp-dev branch as follows:

```
   mkdir $HOME/git/aomp20.0
   cd git/aomp20.0
   git clone https://github.com/ROCm/aomp.git aomp
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
cd $HOME/git/aomp20.0/aomp
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
[AOMP issues] (https://github.com/ROCm/aomp/issues).
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
This bin directory ($HOME/git/aomp20.0/aomp/bin) contains many component scripts to build and install AOMP from source. Here are some of the scripts contained in this direcotry
```
clone_aomp.sh            -  A script to make sure the necessary repos are up to date.
                            See list source repositories above.
build_aomp.sh            -  This is the master build script. It runs all the
                            component build scripts in the correct order.
build_roct.sh            -  Builds the hsa thunk library.
build_rocr.sh            -  Builds the ROCm runtime.
build_project.sh         -  Builds llvm, lld, and clang components.
build_comgr.sh           -  Builds the code object manager (needs rocm-device-libs).
build_rocminfo.sh        -  Builds the rocminfo utilities to support hip.
build_extras.sh          -  Builds hostcall, libm, and utils all in aomp-extras repo.
build_openmp.sh          -  Builds the OpenMP libraries
build_pgmath.sh          -  Builds the pgmath support for flang.
build_flang.sh           -  Builds flang for aomp.
build_flang_runtime.sh   -  Builds the flang runtime for aomp.
build_rocdbgapi.sh       -  Builds ROCm Debugger API.
build_gdb.sh             -  Builds ROC gdb.
build_roctracer.sh       -  Builds ROC gdb. UNDER DEVELOPMENT
build_prereq.sh          -  Builds prerequisites such as cmake, hwloc, rocmsmi in $HOME/local.

create_release_tarball.sh - This builds an important release artifact
                            containing all sources.
```
To build aomp, run the master build script build_aomp.sh, or run the individual scripts in the order shown below.
The build scripts with no arguments will build the component in the build directory $HOME/git/aomp20.0/build/\<component name\>
(or $AOMP_REPOS/build/\<component name\> if AOMP_REPOS is set).
The component build scripts take a single positional argument with the value "install" or "nocmake".
The master build script build_aomp.sh will call the component build scripts in the following order and stop if
any fails occur.

```
   ./build_prereq.sh
   ./build_prereq.sh install
   ./build_project.sh
   ./build_project.sh install
   ./build_roct.sh
   ./build_roct.sh install
   ./build_rocr.sh
   ./build_rocr.sh install
   ./build_openmp.sh
   ./build_openmp.sh install
   ./build_offload.sh
   ./build_offload.sh install
   ./build_extras.sh
   ./build_extras.sh install
   ./build_comgr.sh
   ./build_comgr.sh install
   ./build_rocminfo.sh
   ./build_rocminfo.sh install
   ./build_flang-legacy.sh
   ./build_flang-legacy.sh install
   ./build_pgmath.sh
   ./build_pgmath.sh install
   ./build_flang.sh
   ./build_flang.sh install
   ./build_flang_runtime.sh
   ./build_flang_runtime.sh install
   ./build_hipcc.sh
   ./build_hipcc.sh install
   ./build_hipamd.sh
   ./build_hipamd.sh install
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
The AOMP_REPOS default value is currently $HOME/git/aomp20.0.
The out-of-source build directory for each component is $AOMP_REPOS/build/\<component_name\>.

The install location is defined by the $AOMP environment variable. The value of AOMP MUST be reserved as a symbolic link.
That is, the physical installation will be in directory name formed by concatonating the version string to the value of $AOMP.
The "build_project.sh install" script will make a symbolic link from the physical directory to the symbolic directory $AOMP.
The default value for AOMP is $HOME/rocm/aomp.
For example, when building AOMP version 20.0-0 the install scripts will put all files and directories
in $HOME/rocm/aomp_20.0-0 and create a symbolic link as follows:

```
ln -sf ${AOMP}_20.0-0 ${AOMP}
```
All testing for AOMP uses the environment variable AOMP to locate the installation. This makes it easy to switch between versions of AOMP for testing by simply changing the environment variable AOMP. You do NOT need to change the symbolic link.
For example, if the aomp symbolic link currently points to aomp_20.0-0 and you want to test aomp_19.0-3, do this:

```
export AOMP=$HOME/rocm/aomp_19.0-3
```

The aomp package installs in /usr/lib/aomp_\<version_string\> and symlinks /usr/lib/aomp to the versioned directory. To test the installed package, set AOMP to /usr/lib/aomp or /usr/lib/aomp_\<version_string\>.

It is also possible to test the ROCm compiler with AOMP tests by setting AOMP as follows:

```
export AOMP=/opt/rocm/lib/llvm
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

## AOMP_USE_CCACHE

The default is 1, which creates these CMake vars:

```
-DCMAKE_C_COMPILER_LAUNCHER=$_ccache_bin
-DCMAKE_CXX_COMPILER_LAUNCHER=$_ccache_bin
```

Change it to 0 to avoid `ccache`.


## AOMP_SKIP_FLANG

The default for this variable is 0. Setting this to 1 will shorten build time by
changing the defaults for AOMP_COMPONENT_LIST and AOMP_PROJECTS_LIST to

```
AOMP_COMPONENT_LIST=" prereq project"
AOMP_PROJECTS_LIST="clang;lld"
```

The AOMP_SKIP_FLANG variable is for developers working upstream but not on LLVM flang.
