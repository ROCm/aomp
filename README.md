AOMP - V 0.6-1
==============

AOMP:  AMD OpenMP Compiler

This is README.md for https://github.com/ROCM-Developer-Tools/aomp .  This is the base repository for AOMP,  Use this for issues, documentation, packaging, examples, build.

The last release of AOMP is version 0.6-1.  Currently version 0.6-2 is under development.

AOMP is an experimental PROTOTYPE that is intended to support multiple programming models including OpenMP 4.5+,
, HIP, and cuda clang.  It supports offloading to multiple GPU acceleration targets(multi-target).  It also supports different host platforms such as AMD64, PPC64LE, and AARCH64. (multi-platform). 

The bin directory of this repository contains a README.md and build scripts needed to build and install AOMP. However, we recommend that you install from the debian or rpm packages described below.

Attention Users!  Use this repository for issues. Do not put issues in the source code repositories.  Before creating an issue, you may want to see the developers list of TODOs.  See link below.

Table of contents
-----------------

- [Copyright and Disclaimer](#Copyright)
- [Software License Agreement](LICENSE)
- [Verify and Install Linux Support](#Linux-Support)
- [Install](#Install)
- [Getting Started](#Getting-Started)
- [Examples](examples)
- [Development](bin/README.md)
- [TODOs](bin/TODOs) List of TODOs for this release
- [Limitations](#Limitations)

## Copyright and Disclaimer

<A NAME="Copyright">
Copyright (c) 2019 ADVANCED MICRO DEVICES, INC.

AMD is granting you permission to use this software and documentation (if any) (collectively, the 
Materials) pursuant to the terms and conditions of the Software License Agreement included with the 
Materials.  If you do not have a copy of the Software License Agreement, contact your AMD 
representative for a copy.

You agree that you will not reverse engineer or decompile the Materials, in whole or in part, except for 
example code which is provided in source code form and as allowed by applicable law.

WARRANTY DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY 
KIND.  AMD DISCLAIMS ALL WARRANTIES, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING BUT NOT 
LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE, TITLE, NON-INFRINGEMENT, THAT THE SOFTWARE WILL RUN UNINTERRUPTED OR ERROR-
FREE OR WARRANTIES ARISING FROM CUSTOM OF TRADE OR COURSE OF USAGE.  THE ENTIRE RISK 
ASSOCIATED WITH THE USE OF THE SOFTWARE IS ASSUMED BY YOU.  Some jurisdictions do not 
allow the exclusion of implied warranties, so the above exclusion may not apply to You. 

LIMITATION OF LIABILITY AND INDEMNIFICATION:  AMD AND ITS LICENSORS WILL NOT, 
UNDER ANY CIRCUMSTANCES BE LIABLE TO YOU FOR ANY PUNITIVE, DIRECT, INCIDENTAL, 
INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM USE OF THE SOFTWARE OR THIS 
AGREEMENT EVEN IF AMD AND ITS LICENSORS HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH 
DAMAGES.  In no event shall AMD's total liability to You for all damages, losses, and 
causes of action (whether in contract, tort (including negligence) or otherwise) 
exceed the amount of $100 USD.  You agree to defend, indemnify and hold harmless 
AMD and its licensors, and any of their directors, officers, employees, affiliates or 
agents from and against any and all loss, damage, liability and other expenses 
(including reasonable attorneys' fees), resulting from Your use of the Software or 
violation of the terms and conditions of this Agreement.  

U.S. GOVERNMENT RESTRICTED RIGHTS: The Materials are provided with "RESTRICTED RIGHTS." 
Use, duplication, or disclosure by the Government is subject to the restrictions as set 
forth in FAR 52.227-14 and DFAR252.227-7013, et seq., or its successor.  Use of the 
Materials by the Government constitutes acknowledgement of AMD's proprietary rights in them.

EXPORT RESTRICTIONS: The Materials may be subject to export restrictions as stated in the 
Software License Agreement.

## AOMP Verify Linux Support
<A NAME="Linux-Support">

### Verify and Install Linux Support
AOMP needs certain support for Linux to function properly, such as the KFD driver for AMD GPUs and CUDA for NVIDIA. Click [LINUXSUPPORT](LINUXSUPPORT.md) for more information.

## AOMP Install V 0.6-1

<A NAME="Install">

### Debian/Ubunutu install

On Ubuntu 18.04 LTS (bionic beaver), run these commands:
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/r/aomp_Ubuntu1804_0.6-1_amd64.deb
sudo dpkg -i aomp_Ubuntu1804_0.6-1_amd64.deb
```
The AOMP bin directory (which includes the standard clang and llvm binaries) is not intended to be in your PATH for typical operation.

<!--### RPM Install
For rpm-based Linux distributions, use this rpm
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/r/aomp-0.6-1.x86_64.rpm
sudo rpm -i aomp-0.6-1.x86_64.rpm
```
-->
### No root Debian Install

By default, the packages install their content to the release directory /opt/rocm/aomp_0.X-Y and then a  symbolic link is created at /opt/rocm/aomp to the release directory. This requires root access.

To install the debian package without root access into your home directory, you can run these commands.
```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/r/aomp_Ubuntu1604_0.6-1_amd64.deb
   dpkg -x aomp_Ubuntu1604_0.6-1_amd64.deb /tmp/temproot
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
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/r/aomp-0.6-1.x86_64.rpm
   rpm2cpio aomp-0.6-1.x86_64.rpm | cpio -idmv
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

Run 'make help' for more information.  
```
View the OpenMP Examples [README](/examples/openmp) for more information.

## AOMP Limitations

<A NAME="Limitations">

See the release notes in github.  Here are some limitations. 

```
 - Dwarf debugging is turned off for GPUs. -g will turn on host level debugging only.
 - Some simd constructs fail to vectorize on both host and GPUs.  
 - There are debug versions of the runtime libraries.  However, these use printf on the device
   which currently print when the kernel terminates.  So it is not a very useful debug feature if the GPU 
   kernel crashes. 
```
