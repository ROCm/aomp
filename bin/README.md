AOMP Developer README.md
========================

AOMP is the AMD OpenMP Compiler. This is the AOMP developer README stored at:
```
https://github.com/ROCm-Developer-Tools/aomp/bin/README.md
```
The AOMP compiler supports OpenMP, clang-hip, clang-cuda, device OpenCL, and the offline kernel compilatino tool called cloc.  It contains a recent version of the AMD Lightning compiler (llvm amdgcn backend) and the llvm nvptx backend.  Except for clang-cuda, this compiler works for both Nvidia and AMD Radeon GPUs.

This bin directory contains scripts to build AOMP from source.
```
clone_aomp.sh      -  A script to make sure the necessary repos are up to date.
                      See below for a list of these source repositories.

build_aomp.sh      -  Run all build and install scripts in the correct order.

build_roct.sh      -  Build the hsa thunk library

build_rocr.sh      -  Built the ROCm runtime

build_llvm.sh      -  Build llvm, clang, and lld components of AOMP compiler.

build_utils.sh     -  Builds the AOMP utilities

build_atmi.sh      -  Builds early release of ATMI for aomp.

build_hcc.sh       -  Builds the hcc compiler needed by hip

build_hip.sh       -  Builds the hip host runtimes needed by aomp.

build_openmp.sh    -  Builds the OpenMP libraries for aomp.

build_libdevice.sh -  Builds the device bc libraries from rocm-device-libs

build_libm.sh      -  Built the libm DBCL (Device BC Library)
```
These scripts install into $HOME/rocm/aomp (or $AOMP if set).

The clone_aomp.sh script clones the necessary repositories and the correct
branches into subdirectories of $HOME/git/aomp (or $AOMP_REPOS if set)
The repositories needed by AOMP are shown in the following table.
The first column is the AOMP component that uses the repositories.

| Component | SUBDIRECTORY                          | REPOSITORY LINKS
| --------- | ------------                          | ----------------
| roct      | $HOME/git/aomp/roct-thunk-interfaces  | [roct-thunk-interfaces](https://github.com/radeonopencompute/roct-thunk-interface)
| rocr      | $HOME/git/aomp/rocr-runtime           | [rocr-runtime](https://github.com/radeonopencompute/rocr-runtime)
| llvm      | $HOME/git/aomp/clang                  | [clang](https://github.com/ROCm-Developer-Tools/clang)
| llvm      | $HOME/git/aomp/llvm                   | [llvm](https://github.com/ROCm-Developer-Tools/llvm)
| llvm      | $HOME/git/aomp/lld                    | [lld](https://github.com/ROCm-Developer-Tools/lld)
| utils     | $HOME/git/aomp/aomp                   | [aomp/utils](https://github.com/ROCm-Developer-Tools/aomp/tree/master/utils)
| hcc       | $HOME/git/aomp/hcc                    | [hcc](https://github.com/radeonopencompute/hcc)
| hip       | $HOME/git/aomp/hip                    | [hip](https://github.com/ROCm-Developer-Tools/hip)
| atmi      | $HOME/git/aomp/atmi                   | [atmi](https://github.com/radeonopencompute/atmi)
| openmp    | $HOME/git/aomp/openmp                 | [openmp](https://github.com/ROCm-Developer-Tools/openmp)
| libdevice | $HOME/git/aomp/rocm-device-libs       | [rocm-device-libs](https://github.com/radeonopencompute/rocm-device-libs)
| libm      | $HOME/git/aomp/aomp                   | [aomp/examples](https://github.com/ROCm-Developer-Tools/aomp/tree/master/examples/libdevice)
|           | $HOME/git/aomp/openmpapps             | [openmpapps](https://github.com/AMDComputeLibraries/openmpapps)


The scripts and example makefiles use these environment variables and these
defaults if they are not set. This is not a complete list.  See the script headers
for other environment variables that you may override including repo names.

```
   AOMP              $HOME/rocm/aomp
   CUDA              /usr/local/cuda
   AOMP_REPOS        $HOME/git/aomp
   BUILD_TYPE        Release
```

Many other environment variables can be set. See the file [aomp_common_vars](aomp_common_vars) that is sourced by all build scritps. This file has the names of the branches that make up the release that is in development.  You can override the appropriate environment variable if you want to test your source build with a different release.


You can override the above by setting by setting values in your .bashrc or .bash_profile.
Here is a sample for your .bash_profile

```
SUDO="disable"
AOMP=$HOME/install/aomp
BUILD_TYPE=Debug
NVPTXGPUS=30,35,50,60,70
export SUDO AOMP NVPTXGPUS BUILD_TYPE
```
The build scripts will build from the source directories identified by the
environment variable AOMP_REPOS.

To set alternative installation path for the component INSTALL_<COMPONENT> environment
variable can be used, e.g. INSTALL_openmp

To build all components, first clone aomp repo and checkout the master branch
to build our development repository.

```
   git clone https://github.com/ROCm-Developer-Tools/aomp.git
   git checkout master
```
	
To be sure you have the latest sources from the git repositories, run command.

```
   ./clone_aomp.sh
```

The first time you do this, It could take a long time to clone the repositories.  Subsequent calls will pull the latest updates so you can run clone_aomp.sh anytime to be sure you are on the latest development sources.

WANRING: The script clone_aomp.sh does not pull updates for the aomp repository. You must pull aomp repository manually. So please run "clone_aomp.sh" and "cd $HOME/git/aomp/aomp; git pull" frequently to stay current with aomp development.

The Nvidia CUDA SDK is NOT required for a package install of AOMP. However, to build AOMP from source, you MUST have the Nvidia CUDA SDK version 10 installed because AOMP may be used to build applications for NVIDIA GPUs.  The current default build list of Nvidia subarchs is "30,35,50,60,61,70".  For example, the default list will support application builds with --offload-arch=sm_30 and --offload-arch=sm_60 etc.  This build list can be changed with the NVPTXGPUS environment variable. Set this before running build_aomp.sh.

After you have all the source repositories and you have CUDA and all the dependencies installed,
run this script to build aomp.
```
   ./build_aomp.sh
```
Through extensive use of RPATH, all dynamic runtime libraries that are built by any component of AOMP and then are referenced by another AOMP component will resolve the absolute location within the AOMP installation.  This strategy significantly simplifies the AOMP test matrix.  Libraries that may have been installed by a previous ROCm installation including roct and rocr, will not be used by AOMP.


Developers may update a component and then run these  scripts in the folowing order:

```
   ./build_roct.sh
   ./build_roct.sh install

   ./build_rocr.sh
   ./build_rocr.sh install

   ./build_llvm.sh
   ./build_llvm.sh install

   ./build_utils.sh
   ./build_utils.sh install

   ./build_hcc.sh
   ./build_hcc.sh install

   ./build_hip.sh
   ./build_hip.sh install

   ./build_atmi.sh
   ./build_atmi.sh install

   ./build_openmp.sh
   ./build_openmp.sh install

   ./build_libdevice.sh
   ./build_libdevice.sh install

   ./build_libm.sh
   ./build_libm.sh install
```

For now, run this command for some minor fixups to the install.

```
   ./build_fixup.sh
```
Once you have a successful development build, individual components can be incrementally rebuilt without rebuilding the entire system or the entire component. For example, if you change a file in the clang repository. Run this command to incrementally build llvm, clang, and lld and update your installation.
```
   ./build_llvm.sh install
```
The default out-of-source build directory for each component is $HOME/git/aomp/build/<component>.

WARNING: When the build scripts are run with NO arguments (that is, you do not specify "install" or "nocmake"), the build scripts will rebuild the entire component by DELETING THE BUILD DIRECTORY before running cmake and make.
