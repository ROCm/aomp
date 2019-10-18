AOMP Developer README.md
========================

AOMP is a scripted build of LLVM and supporting software. It has support for OpenMP target offload on amdgcn GPUs.
This is the AOMP developer README stored at:
```
https://github.com/ROCm-Developer-Tools/aomp/blob/master/bin/README.md
```
The AOMP compiler supports OpenMP, clang-hip, clang-cuda, device OpenCL, and the offline kernel compilatino tool called cloc.  It contains a recent version of the AMD Lightning compiler (llvm amdgcn backend) and the llvm nvptx backend.  Except for clang-cuda, this compiler works for both Nvidia and AMD Radeon GPUs.

This bin directory contains scripts to build AOMP from source.
```
clone_aomp.sh            -  A script to make sure the necessary repos are up to date.
                            See below for a list of these source repositories.
build_aomp.sh            -  Run all build and install scripts in the correct order.
build_roct.sh            -  Builds the hsa thunk library.
build_rocr.sh            -  Builds the ROCm runtime.
build_project.sh         -  Builds llvm, lld, and clang components.
build_libdevice.sh       -  Builds the rocdl device bc libraries from rocm-device-libs.
build_comgr.sh           -  Builds the code object manager (needs rocm-device-libs).
build_rocminfo.sh        -  Builds the rocminfo utilities to support hip.
build_hcc.sh             -  Builds the hcc compiler needed by hip.
build_hip.sh             -  Builds the hip host runtimes needed by aomp.
build_extras.sh          -  Builds hostcall, libm, and utils all stored in the aomp-extras repo
build_atmi.sh            -  Builds early release of ATMI for aomp.
build_openmp.sh          -  Builds the OpenMP libraries for aomp.
build_pgmath.sh          -  Builds the pgmath support for flang.
build_flang.sh           -  Builds the flang for aomp.
build_flang_runtime.sh   -  Builds the flang runtime for aomp.
```

These scripts install into $HOME/rocm/aomp (or $AOMP if set).

## Repositories
<A NAME="Repositories">
The clone_aomp.sh script clones the necessary repositories and the correct
branches into subdirectories of $HOME/git/aomp (or $AOMP_REPOS if set)
The repositories needed by AOMP are shown in the following table.
The first column is the AOMP component that uses the repositories.

| Component | SUBDIRECTORY                          | REPOSITORY LINKS
| --------- | ------------                          | ----------------
| roct      | $HOME/git/aomp/roct-thunk-interfaces  | [roct-thunk-interfaces](https://github.com/radeonopencompute/roct-thunk-interface)
| rocr      | $HOME/git/aomp/rocr-runtime           | [rocr-runtime](https://github.com/radeonopencompute/rocr-runtime)
| llvm-project  | $HOME/git/aomp/llvm-project       | [llvm-project](https://github.com/ROCm-Developer-Tools/llvm-project)
| extras    | $HOME/git/aomp/aomp-extras            | [aomp-extras](https://github.com/ROCm-Developer-Tools/aomp-extras)
| hcc       | $HOME/git/aomp/hcc                    | [hcc](https://github.com/radeonopencompute/hcc)
| comgr     | $HOME/git/aomp/rocm-compilersupport   | [comgr](https://github.com/radeonopencompute/rocm-compilersupport)
| hip       | $HOME/git/aomp/hip                    | [hip](https://github.com/ROCm-Developer-Tools/hip)
| atmi      | $HOME/git/aomp/atmi                   | [atmi](https://github.com/radeonopencompute/atmi)
| openmp    | $HOME/git/aomp/llvm-project/openmp    | [llvm-project/openmp](https://github.com/ROCm-Developer-Tools/llvm-project)
| libdevice | $HOME/git/aomp/rocm-device-libs       | [rocm-device-libs](https://github.com/radeonopencompute/rocm-device-libs)
| flang     | $HOME/git/aomp/flang                  | [flang](https://github.com/ROCm-Developer-Tools/flang)
| rocminfo  | $HOME/git/aomp/rocminfo               | [rocminfo](https://github.com/radeonopencompute/rocminfo)
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

Many other environment variables can be set. See the file [aomp_common_vars](aomp_common_vars) that is sourced by all build scripts. This file has the names of the branches that make up the release that is in development.  You can override the appropriate environment variable if you want to test your source build with a different release.


You can override the above by setting by setting values in your .bashrc or .bash_profile.
Here is a sample for your .bash_profile

```
SUDO="disable"
AOMP=$HOME/install/aomp
BUILD_TYPE=Debug
NVPTXGPUS=30,35,50,60,61,70
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

The Nvidia CUDA SDK is NOT required for a package install of AOMP. However, to build AOMP from source, you SHOULD have the Nvidia CUDA SDK version 10 installed because AOMP may be used to build applications for NVIDIA GPUs.  The current default build list of Nvidia subarchs is "30,35,50,60,61,70".  For example, the default list will support application builds with --offload-arch=sm_30 and --offload-arch=sm_60 etc.  This build list can be changed with the NVPTXGPUS environment variable. Set this before running build_aomp.sh.

```
export AOMP_BUILD_CUDA=1 //build AOMP with nvptx support
```

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

   ./build_project.sh
   ./build_project.sh install

   ./build_libdevice.sh
   ./build_libdevice.sh install

   ./build_comgr.sh
   ./build_comgr.sh install

   ./build_rocminfo.sh
   ./build_rocminfo.sh install

   ./build_hcc.sh
   ./build_hcc.sh install

   ./build_hip.sh
   ./build_hip.sh install

   ./build_extras.sh
   ./build_extras.sh install

   ./build_atmi.sh
   ./build_atmi.sh install

   ./build_openmp.sh
   ./build_openmp.sh install

   ./build_pgmath.sh
   ./build_pgmath.sh install

   ./build_flang.sh
   ./build_flang.sh install

   ./build_flang_runtime.sh
   ./build_flang_runtime.sh install
```

For now, run this command for some minor fixups to the install.

```
   ./build_fixups.sh
```
Once you have a successful development build, individual components can be incrementally rebuilt without rebuilding the entire system or the entire component. For example, if you change a file in the llvm-project repository. Run this command to incrementally build llvm, clang, and lld and update your installation.
```
   ./build_project.sh install
```
The default out-of-source build directory for each component is $HOME/git/aomp/build/<component>.

WARNING: When the build scripts are run with NO arguments (that is, you do not specify "install" or "nocmake"), the build scripts will rebuild the entire component by DELETING THE BUILD DIRECTORY before running cmake and make.
