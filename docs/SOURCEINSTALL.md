# Source Install V 22.0-1

Build and install from sources is possible.  However, the source build for AOMP is complex for several reasons.
- Many repos are required.
- Requires that Cuda SDK 10/11 is installed for NVIDIA GPUs. ROCm does not need to be installed for AOMP.
- It is a bootstrapped build. The built and installed LLVM compiler is used to build library components.
- Additional package dependencies are required that are not required when installing the AOMP package.

## Source Build Prerequisites

To build and test AOMP from source you must:
```
1. Install certain distribution packages,
2. Build CMake 3.25.2 from source. This can be done with ./build_prereq.sh,
3. Ensure the KFD kernel module is installed and operating,
4. Create the Unix video group, and
5. Install spack if required.
```
[This link](SOURCEINSTALL_PREREQUISITE.md) provides detailed instructions to satisfy all the AOMP source build requirements.

## Clone and Build AOMP

AOMP now uses a manifest file to specify the git repositories to clone.
We studied the use of the google repo command and found it was too compilicated for development
as compared to manually cloning the repos specified in a manifest file.
The script clone\_aomp.sh issues a sequence of "git clone" and "git pull" commands
by parsing information in the manifest file associated with a particular release.

<b>Choose a Build Version (Development or Release)</b> The development version is the next version to be released. It is possible that the development version is broken due to regressions that often occur during development.

Here are the commands to do a source build of AOMP:

<b>Development Branch:</b>
```
   export AOMP_VERSION=22.0
   export AOMP_REPOS=$HOME/git/aomp${AOMP_VERSION}
   mkdir -p $AOMP_REPOS
   cd $AOMP_REPOS
   git clone -b aomp-dev https://github.com/ROCm-Developer-Tools/aomp
```

The development version is the next version to be released.  It is possible that the development version is broken due to regressions that often occur during development.
These commands will build a previous release of AOMP such as aomp-22.0-0.<br>
<b>Release Branch:</b>
```
   export AOMP_VERSION=22.0
   export AOMP_REPOS=$HOME/git/aomp${AOMP_VERSION}
   mkdir -p $AOMP_REPOS
   cd $AOMP_REPOS
   git clone -b aomp-22.0-0 https://github.com/ROCm-Developer-Tools/aomp
```
<b>Clone and build:</b>
```
   $AOMP_REPOS/aomp/bin/clone_aomp.sh
   $AOMP_REPOS/aomp/bin/build_prereq.sh
   nohup $AOMP_REPOS/aomp/bin/build_aomp.sh &
```

<b>Optional build of math libraries (aomp-hip-libraries):</b>
This build can take a long time especially with multiple GPU ARCHS. The user can limit these by setting ROCMLIBS_GFXLIST=gfx90a or ROCMLIBS_GFXLIST="gfx90a;gfx1010".<br>
A full list of currently supported architectures, ROCMLIBS_GFXLIST, can be found in aomp/bin/aomp_common_vars.
```
   $AOMP_REPOS/aomp/bin/rocmlibs/clone_rocmlibs.sh
   $AOMP_REPOS/aomp/bin/rocmlibs/build_rocmlibs.sh
```

Change the value of AOMP\_REPOS to the directory name where you want to store all the repositories needed for AOMP. All the AOMP repositories will consume more than 12GB. Furthermore, each AOMP component will be built in a subdirectory of $AOMP\_REPOS/build which will consume an additional 6GB. So it is recommened that the directory $AOMP\_REPOS have more than 20GB of free space before beginning. It is recommended that $AOMP\_REPOS name include the value of AOMP\_VERSION as shown above. It is also recommended to put the values of AOMP\_VERSION and AOMP\_REPOS in a login profile (such as .bashrc) so future incremental build scripts will correctly find your sources.

Warning: the clone\_aomp.sh, and build\_aomp.sh are expected to take a long time to execute. As such we recommend the use of nohup to run build\_aomp.sh. It is ok to run build\_aomp.sh without nohup. The clone and build time will be affected by the performance of the filesystem that contains $AOMP\_REPOS.

There is a "list" option on the clone\_aomp.sh that provides useful information about each AOMP repository.
```
   $AOMP_REPOS/aomp/bin/clone_aomp.sh list
```
The above command will produce output like this showing you the location and branch of the repos in the AOMP\_REPOS directory and if there are any discrepencies with respect to the manifest file.<br>

<b>USED manifest file: /work/grodgers/git/aomp22.0/aomp/bin/../manifests/aompi_22.0.xml</b><br>
```
  repo src       branch                 path                 repo name    last hash    updated           commitor         for author
  --------       ------                 ----                 ---------    ---------    -------           --------         ----------
       emu    amd-staging          llvm-project              llvm-project 295b2c16cf3e 2025-07-25             GitHub     Lambert, Jacob
       emu    amd-staging SPIRV-LLVM-Translator     spirv-llvm-translator 735c75e91ea4 2025-07-23             GitHub Zhuravlyov, Konstantin
       emu    amd-staging                hipify                    hipify 8198c42a05d1 2025-07-23             GitHub Zhuravlyov, Konstantin
       roc       aomp-dev                  aomp                      aomp 2d3d7e32df30 2025-07-25      Ethan Stewart      Ethan Stewart
       roc   rocm-rel-6.4       rocprofiler-sdk           rocprofiler-sdk e8e49fe76971 2025-03-25             GitHub Mallya, Ameya Keshava
       roc   rocm-rel-6.4             ROCdbgapi                 ROCdbgapi 59be7ff0aaaf 2025-03-31       Lancelot SIX       Lancelot SIX
       roc   rocm-rel-6.4                ROCgdb                    ROCgdb 401bb21f2f3c 2025-03-31       Lancelot SIX       Lancelot SIX
       roc   rocm-rel-6.4                   hip                       hip 3d568a1ba58c 2025-07-07             GitHub    rocm_devops, a1
       roc   rocm-rel-6.4                   clr                       clr 123eb5128769 2025-07-07             GitHub    rocm_devops, a1
       roc   rocm-rel-6.4              rocminfo                  rocminfo 6ea2ba38c8e1 2025-02-17             GitHub   Choudhary, Rahul
       roc   rocm-rel-6.4          rocm_smi_lib              rocm_smi_lib 1603a3281d44 2025-06-12 Harrymanoharan, Jessey Ranjith Ramakrishnan
       roc   rocm-rel-6.4                amdsmi                    amdsmi 41065ee69f01 2025-07-07             GitHub    rocm_devops, a1
       roc   rocm-rel-6.4            rocm-cmake                rocm-cmake ecc716b97c22 2024-11-30             GitHub      Paul Fultz II
       roc   rocm-rel-6.4          rocr-runtime              ROCR-Runtime 044c4226baf2 2025-06-12 Harrymanoharan, Jessey         Eric Huang
       roc   rocm-rel-6.4  rocprofiler-register      rocprofiler-register 7c6cd44f637d 2025-02-17             GitHub   Choudhary, Rahul
       roc   rocm-rel-6.4               hipfort                   hipfort af63249d5e29 2025-06-04             GitHub ROCm CI Service Account

```
For more information, or if you are interested in joining the development of AOMP, please read the AOMP developers README file located here [README](../bin/README.md).
