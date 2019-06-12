# AOMP Install 


## Verify and Install Linux Support for AOMP

<A NAME="Linux-Support">

AOMP needs certain support for Linux to function properly, such as the KFD driver for AMD GPUs and CUDA for nvptx. Click [LINUXSUPPORT](LINUXSUPPORT.md) for more information.

## AOMP Install V 0.6-3

<A NAME="Install">

### Debian/Ubunutu Install

On Ubuntu 18.04 LTS (bionic beaver), run these commands:
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.6-3/aomp_Ubuntu1804_0.6-3_amd64.deb
sudo dpkg -i aomp_Ubuntu1804_0.6-3_amd64.deb
```
The AOMP bin directory (which includes the standard clang and llvm binaries) is not intended to be in your PATH for typical operation.

<!--### RPM Install
For rpm-based Linux distributions, use this rpm
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/r/aomp-0.6-3.x86_64.rpm
sudo rpm -i aomp-0.6-3.x86_64.rpm
```
-->
### No root Debian Install

By default, the packages install their content to the release directory /opt/rocm/aomp_0.X-Y and then a  symbolic link is created at /opt/rocm/aomp to the release directory. This requires root access.

To install the debian package without root access into your home directory, you can run these commands.
```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.6-3/aomp_Ubuntu1604_0.6-3_amd64.deb
   dpkg -x aomp_Ubuntu1604_0.6-3_amd64.deb /tmp/temproot
   mv /tmp/temproot/opt/rocm $HOME
   export PATH=$PATH:$HOME/rocm/aomp/bin
   export AOMP=$HOME/rocm/aomp
```
The last two commads could be put into your .bash_profile file so you can always access the compiler.

### No root RPM install

By default, the packages install their content to the release directory /opt/rocm/aomp_0.X-Y and then a  symbolic link is created at /opt/rocm/aomp to the release directory. This requires root access.

To install the rpm package without root access into your home directory, you can run these commands.
```
   mkdir /tmp/temproot ; cd /tmp/temproot 
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.6-3/aomp-0.6-3.x86_64.rpm
   rpm2cpio aomp-0.6-3.x86_64.rpm | cpio -idmv
   mv /tmp/temproot/opt/rocm $HOME
   export PATH=$PATH:$HOME/rocm/aomp/bin
   export AOMP=$HOME/rocm/aomp
```
The last two commads could be put into your .bash_profile file so you can always access the compiler.

### Source Install
Build and install from sources is possible.  However, the source build for AOMP is complex for several reasons.
- Many repos are required.  The clone_aomp.sh script ensures you have all repos and the correct branch.
- Requires that Cuda SDK 10 is installed for NVIDIA GPUs. ROCm does not need to be installed for AOMP.
- It is a bootstrapped build. The built and installed LLVM compiler is used to build library components.
- Additional package dependencies are required that are not required when installing the AOMP package.

For instructions on how to install from source, click [SOURCEINSTALL](SOURCEINSTALL.md).

## Getting Started

<A NAME="Getting-Started">

The default install location is /opt/rocm/aomp. To run the given examples, for example in /opt/rocm/aomp/examples/openmp do the following:

### Copy the example openmp directory somewhere writable
```
cd /opt/rocm/aomp/examples/
cp -rp openmp /work/tmp/openmp-examples
cd /work/tmp/openmp-examples/vmulsum
```

### Point to the installed AOMP by setting AOMP environment variable
```
export AOMP=/opt/rocm/aomp
```

### Make Instructions
```
make clean
make run
```
Run 'make help' for more details.  

View the OpenMP Examples [README](../examples/openmp) for more information.
