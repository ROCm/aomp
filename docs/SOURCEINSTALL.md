# Source Install V 16.0-2

Build and install from sources is possible.  However, the source build for AOMP is complex for several reasons.
- Many repos are required.
- Requires that Cuda SDK 10 is installed for NVIDIA GPUs. ROCm does not need to be installed for AOMP.
- It is a bootstrapped build. The built and installed LLVM compiler is used to build library components.
- Additional package dependencies are required that are not required when installing the AOMP package.

## Source Build Prerequisites

To build AOMP from source you must: 1. Install certain distribution packages, 2. Build CMake 3.13.4 from source 3. ensure the KFD kernel module is installed and operating, 4. create the Unix video group, and 5. install spack if required. [This link](SOURCEINSTALL_PREREQUISITE.md) provides instructions to satisfy all the AOMP source build dependencies.

## Clone and Build AOMP

```
  repo src       branch                 path                 repo name    last hash    updated           commitor         for author
  --------       ------                 ----                 ---------    ---------    -------           --------         ----------
 gerritgit amd-stg-open         llvm-project lightning/ec/llvm-project 11d5fa11d52c 2021-10-30      Ron Lieberman     Saiyedul Islam
       roc amd-stg-open rocm-compilersupport      ROCm-CompilerSupport ef72f4edb3dd 2021-10-12      Jeremy Newton      Jeremy Newton
       roc amd-stg-open     rocm-device-libs          ROCm-Device-Libs fc2eac750cb3 2021-10-26           pvellien           pvellien
  roctools     aomp-dev                flang                     flang 4f1282c59a76 2021-09-22             GitHub            ronlieb
  roctools     aomp-dev          aomp-extras               aomp-extras aefc0d6c7434 2021-10-18      Ron Lieberman      Ron Lieberman
  roctools     aomp-dev                 aomp                      aomp a94d9f43f3b0 2021-11-01        gregrodgers        gregrodgers
  roctools   rocm-5.2.x          rocprofiler               rocprofiler e140f47f3609 2021-10-08 Gerrit Code Review      Zhongyu Zhang
  roctools   rocm-5.2.x            roctracer                 roctracer d8ecefda4efd 2021-10-09      Ammar Elwazir      Ammar ELWazir
  roctools   rocm-5.2.x            ROCdbgapi                 ROCdbgapi 1040f1521831 2021-09-26 Laurent Morichetti Laurent Morichetti
  roctools   rocm-5.2.x               ROCgdb                    ROCgdb a1f2a479f060 2021-09-24 Laurent Morichetti       Simon Marchi
  roctools   rocm-5.2.x               hipamd                    hipamd bedc5f614221 2021-10-04  Pruthvi Madugundu            Sourabh
  roctools   rocm-5.2.x                  hip                       hip 3413a164f458 2021-10-11      Maneesh Gupta      Maneesh Gupta
  roctools   rocm-5.2.x               ROCclr                    ROCclr aba55f5c2775 2021-10-07      Zhongyu Zhang   German Andryeyev
       roc   rocm-5.2.x  ROCm-OpenCL-Runtime       ROCm-OpenCL-Runtime bf77cab71234 2021-09-27        Freddy Paul        Freddy Paul
       roc    roc-5.2.x             rocminfo                  rocminfo 1452f8fa24b2 2021-07-19      Icarus Sparry      Icarus Sparry
       roc rocm-rel-5.2           rocm-cmake                rocm-cmake 8d82398d269d 2021-09-14            Jenkins            Jenkins
       roc   rocm-5.2.x         rocr-runtime              ROCR-Runtime f32dfa887d02 2021-10-27         Sean Keely         Sean Keely
       roc    roc-5.2.x roct-thunk-interface      ROCT-Thunk-Interface 5b152ed0f043 2021-09-30         Sean Keely         Sean Keely
     rocsw rocm-rel-5.2              hipfort                   hipfort 96e9cb88281d 2021-11-16             GitHub   Dominic Charrier
```
For more information, or if you are interested in joining the development of AOMP, please read the AOMP developers README file located here [README](../bin/README.md).
