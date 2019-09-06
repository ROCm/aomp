AOMP - V 0.7-1
==============

AOMP:  AMD OpenMP Compiler

This is README.md for https://github.com/ROCM-Developer-Tools/aomp .  This is the base repository for AOMP. Use this for issues, documentation, packaging, examples, and building.

The last release of AOMP is version 0.7-1.  Currently version 0.7-2 is under development.

AOMP is a scripted build of clang/llvm.  It has support for OpenMP target offload on amdgcn GPUs.
Some source to support OpenMP target offload on amdgcn GPUs has not yet been merged
into the upstream clang/llvm trunk.
But all of the source used by AOMP is available in AOMP repositories as a set of commits on top of the clang/llvm github mono-repo.
Since AOMP is a clang/llvm compiler, it supports GPU offloading with HIP, CUDA, OpenCL, and OpenMP.

The bin directory of this repository contains a README.md and build scripts needed to download, build, and install AOMP from source.
In addition to the  mirrored and modified llvm repository (llvm-project), AOMP uses a number of open-source ROCm components. The build scripts will download, build, and install all components needed for AOMP.
However, we recommend that you install the latest release of the debian or rpm package for AOMP described in the [install page](docs/INSTALL.md).

Attention Users!  Please use this repository for issues. Do not put issues in the source code repositories.

Table of contents
-----------------

- [Copyright and Disclaimer](#Copyright)
- [Software License Agreement](LICENSE)
- [Install](docs/INSTALL.md)
- [Test Install](docs/TESTINSTALL.md)
- [Examples](examples)
- [Development](bin/README.md)
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

## AOMP Limitations

<A NAME="Limitations">

See the release notes in github.  Here are some limitations. 

```
 - Dwarf debugging is turned off for GPUs. -g will turn on host level debugging only.
 - Some simd constructs fail to vectorize on both host and GPUs.  
```
