AOMP Developer README.md
========================

AOMP is a scripted build of LLVM and supporting software. It has support for OpenMP target offload on amdgcn GPUs.
This is the AOMP developer README stored at:
```
https://github.com/ROCm-Developer-Tools/aomp/blob/master/bin/README.md
```
The AOMP compiler supports OpenMP, clang-hip, clang-cuda, device OpenCL, and the offline kernel compilation tool called cloc.  It contains a recent version of the AMD Lightning compiler (llvm amdgcn backend) and the llvm nvptx backend.  Except for clang-cuda, this compiler works for both Nvidia and AMD Radeon GPUs.

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
build_vdi.sh             -  Builds the hip interoperability layer ROCclr.
build_hipvdi.sh          -  Builds the hip host runtime using ROCclr.
build_extras.sh          -  Builds hostcall, libm, and utils all in aomp-extras repo.
build_openmp.sh          -  Builds the OpenMP libraries and device RTL.
build_pgmath.sh          -  Builds the pgmath support for flang.
build_flang.sh           -  Builds flang for aomp.
build_flang_runtime.sh   -  Builds the flang runtime for aomp.
```

These scripts install into $HOME/rocm/aomp (or $AOMP if set).

## Repositories
<A NAME="Repositories">
The clone_aomp.sh script clones the necessary repositories and the correct
branches into subdirectories of $HOME/git/aomp11 (or $AOMP_REPOS if set)
The repositories needed by AOMP are shown in the following table.
The first column is the AOMP component that uses the repositories.

| Component | SUBDIRECTORY                           | REPOSITORY LINKS
| --------- | ------------                           | ----------------
| roct      | $HOME/git/aomp11/roct-thunk-interfaces | [roct-thunk-interfaces](https://github.com/radeonopencompute/roct-thunk-interface)
| rocr      | $HOME/git/aomp11/rocr-runtime          | [rocr-runtime](https://github.com/radeonopencompute/rocr-runtime)
| llvm-project | $HOME/git/aomp11/amd-llvm-project      | [llvm-project](https://github.com/ROCm-Developer-Tools/amd-llvm-project)
| extras    | $HOME/git/aomp11/aomp-extras           | [aomp-extras](https://github.com/ROCm-Developer-Tools/aomp-extras)
| vdi       | $HOME/git/aomp11/vdi                   | [vdi](https://github.com/ROCm-Developer-Tools/ROCclr)
| comgr     | $HOME/git/aomp11/rocm-compilersupport  | [comgr](https://github.com/radeonopencompute/rocm-compilersupport)
| hipvdi    | $HOME/git/aomp11/hip-on-vdi            | [hipvdi](https://github.com/ROCm-Developer-Tools/hip)
| ocl       | $HOME/git/aomp11/opencl-on-vdi         | [ocl](https://github.com/radeonopencompute/ROCm-OpenCL-Runtime)
| openmp    | $HOME/git/aomp11/llvm-project/openmp   | [llvm-project/openmp](https://github.com/ROCm-Developer-Tools/llvm-project)
| libdevice | $HOME/git/aomp11/rocm-device-libs      | [rocm-device-libs](https://github.com/radeonopencompute/rocm-device-libs)
| flang     | $HOME/git/aomp11/flang                 | [flang](https://github.com/ROCm-Developer-Tools/flang)
| rocminfo  | $HOME/git/aomp11/rocminfo              | [rocminfo](https://github.com/radeonopencompute/rocminfo)

The scripts and example makefiles use these environment variables and these
defaults if they are not set. This is not a complete list.  See the script headers
for other environment variables that you may override including repo names.

```
   AOMP              $HOME/rocm/aomp
   CUDA              /usr/local/cuda
   AOMP_REPOS        $HOME/git/aomp11
   BUILD_TYPE        Release
```

Many other environment variables can be set. See the file [aomp_common_vars](aomp_common_vars) that is sourced by all build scripts. This file sets default values for branch names and repo directory name.

You can override the above by setting by setting values in your .bashrc or .bash_profile.
Here is a sample of commands you might want to put into your .bash_profile:
```
AOMP=$HOME/rocm/aomp
BUILD_TYPE=Debug
NVPTXGPUS=30,35,50,60,61,70
export AOMP NVPTXGPUS BUILD_TYPE
```
To build and clone all components using the latest development sources, first clone aomp repo and checkout the amd-stg-openmp branch as follows:

```
   mkdir $HOME/git
   cd git
   git clone https://github.com/ROCm-Developer-Tools/aomp.git aomp11
   cd aomp11
   git checkout amd-stg-openmp
   ./clone_aomp.sh
```
The first time you run ./clone_aomp.sh, it could take a long time to clone the repositories.
Subsequent calls will only pull the latest updates.

WARNING: The script clone_aomp.sh does not pull updates for this aomp repository. You must pull aomp repository manually.
So please run this frequently to stay current with the aomp development team.

```
cd $HOME/git/aomp11/aomp 
git pull
./clone_aomp.sh
```
The Nvidia CUDA SDK is NOT required for a package install of AOMP. However, to build AOMP from source, you SHOULD have the Nvidia CUDA SDK version 10 installed because AOMP may be used to build applications for NVIDIA GPUs. The current default build list of Nvidia subarchs is "30,35,50,60,61,70".  For example, the default list will support application builds with --offload-arch=sm_30 and --offload-arch=sm_60 etc.  This build list can be changed with the NVPTXGPUS environment variable as shown above. 

After you have all the source repositories and you have CUDA and all the dependencies installed,
run this script to build aomp.
```
   ./build_aomp.sh
```
The AOMP source build is a standalone build of all components needed for compilation and execution with the exception of the Linux kernel module. Through extensive use of RPATH, all dynamic runtime libraries that are built by any component of AOMP and referenced by another AOMP component will resolve the absolute location within the AOMP source installation (default is $HOME/rocm/aomp). This strategy significantly simplifies the AOMP test matrix. Libraries that may have been installed by a previous ROCm installation including roct and rocr, will not be used by the source build of AOMP. Nor will the AOMP installation for the source build affect any ROCm component. 

Starting with ROCM 3.0 there is a version of AOMP distributed with ROCm. Unlike the AOMP source build, ROCm AOMP is integrated into ROCm.  That version depends on existing ROCm components. The ROCm AOMP package installs into /opt/rocm/aomp. The default source build installs into $HOME/rocm/aomp.

Developers may update a component and then run these scripts in the following order:

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

   ./build_vdi.sh
   ./build_vdi.sh install

   ./build_hipvdi.sh
   ./build_hipvdi.sh install

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
```
Once you have a successful development build, individual components can be incrementally rebuilt without rebuilding the entire system or the entire component. For example, if you change a file in the llvm-project repository. Run this command to incrementally build llvm, clang, and lld and update your installation.
```
   ./build_project.sh install
```
The build scripts will build from the source directories identified by the environment variable AOMP_REPOS.
The default out-of-source build directory for each component is $HOME/git/aomp11/build/<component>.

WARNING: When the build scripts are run with NO arguments (that is, you do not specify "install" or "nocmake"), the build scripts will rebuild the entire component by DELETING THE BUILD DIRECTORY before running cmake and make. 

The install location is defined by the $AOMP environment variable. The value of AOMP needs to be reserved as a symbolic link.
That is, the physical installation will be in directory $AOMP concatonated with the version string.
The install script will make a symbolic link from the physical directory to the symbolic directory $AOMP.
For example, when building AOMP version 11.6-1 with the default value for AOMP=$HOME/rocm/aomp,
the install scripts will put all files and directories in $HOME/rocm/aomp_11.6-1 and create a symbolic link as follows.

```
ln -sf ${AOMP}_11.6-1 ${AOMP}
```
This symbolic link makes it easy to switch between versions of AOMP for testing.
