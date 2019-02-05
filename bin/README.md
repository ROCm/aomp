AOMP Developer README.md
========================

AOMP is the AMD OpenMP Compiler. This is the AOMP developer README stored at:
```
https://github.com/ROCm-Developer-Tools/aomp/bin/README.md
```
This bin directory contains scripts to build AOMP from source.
```
clone_aomp.sh      -  A script to make sure the necessary repos are up to date.
                      See below for a list of these libraries.

build_aomp.sh      -  Run all build and install scripts in the correct order. 

build_roct.sh      -  Build the hsa thunk library

build_rocr.sh      -  Built the ROCm runtime

build_llvm.sh      -  Build llvm, clang, and lld components of AOMP compiler.
                      This compiler supports openmp, clang hip, clang cuda,
                      opencl, and the GPU  kernel frontend cloc.
                      It contains a recent version of the AMD Lightning compiler
                      (llvm amdgcn backend) and the llvm nvptx backend.
                      This compiler works for both Nvidia and AMD Radeon GPUs.

build_utils.sh     -  Builds the AOMP utilities

build_atmi.sh      -  Builds early release of ATMI for aomp.

build_hip.sh       -  Builds the hip host runtimes needed by aomp.

build_openmp.sh    -  Builds the OpenMP libraries for aomp.

build_libdevice.sh -  Builds the device bc libraries from rocm-device-libs

build_libm.sh      -  Built the libm DBCL (Device BC Library)
```
These scripts install into $HOME/rocm/aomp (or $AOMP if set). 
The repositories needed by AOMP are:

```
DIRECTORY NAME *                  	AOMP REPOSITORY **       
-------------------------------   	---------------------------
$HOME/git/aomp/aomp               	%rocdev/aomp ***
$HOME/git/aomp/clang              	%rocdev/clang ***
$HOME/git/aomp/llvm               	%rocdev/llvm
$HOME/git/aomp/lld                	%rocdev/lld
$HOME/git/aomp/openmp             	%rocdev/openmp ***
$HOME/git/aomp/hip                	%rocdev/hip
$HOME/git/aomp/rocm-device-libs   	%roc/rocm-device-libs
$HOME/git/aomp/atmi               	%roc/atmi
$HOME/git/aomp/rocr-runtime       	%roc/rocr-runtime
$HOME/git/aomp/roct-thunk-interfaces	%roc/roct-thunk-interfaces  master
$HOME/git/aomp/openmpapps         	%roclib/openmpapps

   * Clone your repositories here or override with environment variables.
  ** Replace %roc with "https://github.com/RadeonOpenCompute"
  ** Replace %rocdev with "https://github.com/ROCm-Developer-Tools"
  ** Replace %roclib with "https://github.com/AMDComputeLibraries"
 *** These are the primary development repositories for AOMP. They are updated often.
```

The scripts and example makefiles use these environment variables and these 
defaults if they are not set. This is not a complete list.  See the script headers
for other environment variables that you may override including repo names. 

```
   AOMP              $HOME/rocm/aomp
   CUDA              /usr/local/cuda
   AOMP_REPOS        $HOME/git/aomp
   BUILD_TYPE        Release
```

Many other environment variables can be set.  See the file aomp_common_vars that is sourced by all build scritps. 


You can override the above by setting by setting values in your .bashrc or .bash_profile.
Here is a sample for your .bash_profile

```
SUDO="disable"
AOMP=$HOME/install/aomp
BUILD_TYPE=Debug
NVPTXGPUS=30,35,50,60,70
export SUDO AOMP NVPTXGPUS BUILD_TYPE
```
The sm_70 (70 in NVPTXGPUS) requires CUDA 9 and above.

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

WARING: The script clone_aomp.sh does not pull updates for the aomp repository. You must pull aomp repository manually. 

To build aomp, you MUST have the Nvidia CUDA SDK version 8 or greater installed because AOMP can build applications for NVIDIA GPUs. We have not done testing for CUDA version 10.  The current default list of Nvidia subarchs is "30,35,50,60,70".  For example, the default list will support application builds with --offload-arch=sm_30 and --offload-arch=sm_60 etc.  This list can be changed with the NVPTXGPUS environment variable.

After you have all the source repositories and you have cuda and all the dependencies installed, 
run this script to build aomp.
```
   ./build_aomp.sh
```
The default build directory for each component is $HOME/git/aomp/build/<component>. 

Use build_aomp.sh script if you are confident it will will build without failure.
Otherwise, run these  scripts in the folowing order:

```
   ./build_roct.sh
   ./build_roct.sh install

   ./build_rocr.sh
   ./build_rocr.sh install

   ./build_llvm.sh
   ./build_llvm.sh install

   ./build_utils.sh
   ./build_utils.sh install

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

Once you have a successful development build, components can be incrementally rebuilt without rebuilding the entire system. For example, if you change a file in the clang repository. Run this command to incrementally build llvm, clang, and lld and update your installation. 

```
   ./build_llvm.sh install
```
WARNING:  if you do not specify the "install" option, the script will rebuild the entire component by wiping out the build directory.
