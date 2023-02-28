# How to Build the _trunk17_ Compiler

Add the following to your environment in either `.bash_profile` or `.bashrc`:

``` bash
export TRUNK_REPOS=$HOME/git/trunk17.0
```

Or set `TRUNK_REPOS` to a location with a fast file system. Then log in again
and execute these commands:

``` bash
mkdir -p $TRUNK_REPOS
cd $TRUNK_REPOS
git clone https://github.com/ROCm-Developer-Tools/aomp
git clone https://github.com/ROCm-Developer-Tools/llvm-project
cd llvm-project
git checkout amd-trunk-dev
$TRUNK_REPOS/aomp/trunk/build_trunk.sh 
```


# Environment Variables to Control the Build Process

Default environment variables can be found in the file
`TRUNK_REPOS/aomp/bin/trunk_common_vars`. Do not change this file unless you
are going to commit your changes for everyone. `trunk_common_vars` is designed
to allow you to change the variables in your environment with `.bashrc` or
`.bash_profile`. Next are presented a few of the variables you may want to
override.


## TRUNK_REPOS

See the discussion above for building _trunk17_ on the use of `TRUNK_REPOS`.


## TRUNK_LINK

The default value for `TRUNK_LINK` is `"$HOME/rocm/trunk"`. The `build_trunk.sh`
script will install into directory `${TRUNK_LINK}_17.0-0`.

Furthermore, it creates a symbolic link from `$TRUNK_LINK` to the install dir.
So `$TRUNK_LINK` **MUST NOT** be a physical directory. This symbolic link makes
it easy to switch between future qualifed releases of _trunk17_. If you are on a
system where `$HOME` is in a slow filesystem, set `TRUNK_LINK` to where you want
the install directory to be. For example set the following in your
`.bash_profile` or `.bashrc` then relogin:

``` bash
export TRUNK_LINK=/work/$USER/rocm/trunk
```

Then the install scripts will install into `/work/$USER/rocm/trunk_17.0-0` and
a symlink will be created from `$TRUNK_LINK`.


## NVPTXGPUS

A comma-separated list of CUDA compute capabilities. Use this to reduce build
times and artifacts in the install directories. For example:

``` bash
export NVPTXGPUS='50,70'
```

Will build only for `sm_50` and `sm_70`. Otherwise it will build for all nvidia
GPUs set in `llvm-project/openmp/libomptarget/DeviceRTL/CMakeLists.txt`:

```
set(all_capabilities 35 37 50 52 53 60 61 62 70 72 75 80 86 89 90)
```


## GFXLIST

A space-separated list of AMD GPUs. For example, use this for most of the focus
AMDGPUs. For example:

``` bash
export GFXLIST="gfx906 gfx908 gfx90a"
```

Otherwise, see `llvm-project/openmp/libomptarget/DeviceRTL/CMakeLists.txt`:

```
set(amdgpu_mcpus gfx700 gfx701 gfx801 gfx803 gfx900 gfx902 gfx906 gfx908 gfx90a gfx90c gfx940 gfx1010 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103)
```


## AOMP_USE_CCACHE

The default is 1, which creates these CMake vars:

```
-DCMAKE_C_COMPILER_LAUNCHER=$_ccache_bin
-DCMAKE_CXX_COMPILER_LAUNCHER=$_ccache_bin
```

Change it to 0 to avoid `ccache`.


## TRUNK_BUILD_DEBUG

The default is 0 (up for discussion). If you set `TRUNK_BUILD_DEBUG=1`, then
this CMake variable is added:

```
-DLIBOMPTARGET_ENABLE_DEBUG=ON
```
This is a nice thing to turn on for debugging libomptarget runtime.
Maybe the default should be 1.


## TRUNK_COMPONENTS_LIST

This is a list of components to build.

The default is "prereq project compiler-rt flang"

The build_trunk.sh script will call build_<component_name>.sh twice for each component.
The first time has no arguments and builds without installation. The 2nd time is with the
install argument to install the component.

## TRUNK_PROJECTS_LIST
This is the list of LLVM projects that the build_project.sh script will attempt to build.

The default is "clang;lld;mlir"

Do not add compiler-rt or flang to this list because those parts of LLVM are built with
build_compiler-rt.sh and build_flang.sh because they use the compiler installed by
build_project.sh.

## TRUNK_SKIP_FLANG

The default for this variable is 0. Setting this to 1 will shorten build time by
changing the defaults for TRUNK_COMPONENT_LIST and TRUNK_PROJECTS_LIST to

```
TRUNK_COMPONENT_LIST=" prereq project compiler-rt"
TRUNK_PROJECTS_LIST="clang;lld"
```

The TRUNK_SKIP_FLANG variable is for developers working upstream but not on LLVM flang.

## BUILD_TYPE

The default for `BUILD_TYPE` is "Release". This sets the value for
`CMAKE_BUILD_TYPE`. See CMake documentation for different possible values.


# Releases of _trunk17_

At various development check points we will qualify releases of _trunk17_ and
increment the development version in `trunk_common_vars`. For example, after the
release of `trunk_17.0-0`, development will move to `trunk_17.0-1`. The build
scripts will then install into directory `$HOME/rocm/trunk_17.0-1`. The symbolic
link from `$HOME/rocm/trunk` will also change to the new install directory.

After a _trunk17_ release, a static release branch will be created. such as
`trunk_17.0-0`. This branch will be created by interactive rebasing all the
local commits in `amd-trunk-dev`. Development will continue in the branch
`amd-trunk-dev`.


# Updates From Other Team Members

In the future, there will be a new `clone_trunk.sh` script that will clone or
pull updates on all the repos in `$TRUNK_REPOS` except AOMP. This will be
similar to the `clone_aomp.sh` script used for AOMP. For now, simply run
`git pull` in the repository directory to get updates from other team members or
merges from main.


# Testing _trunk17_
 
To use the various AOMP testing infrastructure in `$TRUNK_REPOS/aomp/test`:

``` bash
export AOMP=$HOME/rocm/trunk
export LD_LIBRARY_PATH=$AOMP/lib:/opt/rocm/hsa/lib
```

Or, if you set `TRUNK_LINK`, use this command:

``` bash
export AOMP=$TRUNK_LINK
export LD_LIBRARY_PATH=$AOMP/lib:/opt/rocm/hsa/lib
```

Then, for example:

``` bash
cd $TRUNK_REPOS/aomp/test/smoke/vmuldemo
make run
```


# MERGE FROM MAIN PROCESS

At any time during the development, the development branch `amd-trunk-dev` can
be updated from `main` by any team member with the script `merge_from_main.sh`.
This script carefully runs the git commands to merge from `main`.

This script ensures that neither the `main` branch nor `amd-trunk-dev` branch has
local changes. Save any of your local changes before using `merge_from_main.sh`.
Note: the `main` branch of
[ROCm-Developer-Tools/llvm-project](https://github.com/ROCm-Developer-Tools/llvm-project)
is automatically mirrored from
[LLVM trunk](https://github.com/llvm/llvm-project) `main` branch every three
hours.

Before running `merge_from_main.sh`, ensure `amd-trunk-dev` is indeed behind
`main` by looking at this
[URL](https://github.com/ROCm-Developer-Tools/llvm-project/branches) or by
running this command:

``` bash
git rev-list --left-right --count main...amd-trunk-dev
```

The script terminates if it finds that `amd-trunk-dev` is 0 commits behind
`main`.
