# Source Install V 17.0-3

Build and install from sources is possible.  However, the source build for AOMP is complex for several reasons.
- Many repos are required.
- Requires that Cuda SDK 10 is installed for NVIDIA GPUs. ROCm does not need to be installed for AOMP.
- It is a bootstrapped build. The built and installed LLVM compiler is used to build library components.
- Additional package dependencies are required that are not required when installing the AOMP package.

## Source Build Prerequisites

To build and test AOMP from source you must:
```
1. Install certain distribution packages,
2. Build CMake 3.22.1 from source. This can be done with ./build_prereq.sh,
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
   export AOMP_VERSION=17.0
   export AOMP_REPOS=$HOME/git/aomp${AOMP_VERSION}
   mkdir -p $AOMP_REPOS
   cd $AOMP_REPOS
   git clone -b aomp-dev https://github.com/ROCm-Developer-Tools/aomp
```

The development version is the next version to be released.  It is possible that the development version is broken due to regressions that often occur during development.
These commands will build a previous release of AOMP such as aomp-17.0-2.<br>
<b>Release Branch:</b>
```
   export AOMP_VERSION=17.0
   export AOMP_REPOS=$HOME/git/aomp${AOMP_VERSION}
   mkdir -p $AOMP_REPOS
   cd $AOMP_REPOS
   git clone -b aomp-17.0-2 https://github.com/ROCm-Developer-Tools/aomp
```
<b>Clone and build:</b>
```
   $AOMP_REPOS/aomp/bin/clone_aomp.sh
   $AOMP_REPOS/aomp/bin/build_prereq.sh
   nohup $AOMP_REPOS/aomp/bin/build_aomp.sh &
```

Change the value of AOMP\_REPOS to the directory name where you want to store all the repositories needed for AOMP. All the AOMP repositories will consume more than 12GB. Furthermore, each AOMP component will be built in a subdirectory of $AOMP\_REPOS/build which will consume an additional 6GB. So it is recommened that the directory $AOMP\_REPOS have more than 20GB of free space before beginning. It is recommended that $AOMP\_REPOS name include the value of AOMP\_VERSION as shown above. It is also recommended to put the values of AOMP\_VERSION and AOMP\_REPOS in a login profile (such as .bashrc) so future incremental build scripts will correctly find your sources.

Warning: the clone\_aomp.sh, and build\_aomp.sh are expected to take a long time to execute. As such we recommend the use of nohup to run build\_aomp.sh. It is ok to run build\_aomp.sh without nohup. The clone and build time will be affected by the performance of the filesystem that contains $AOMP\_REPOS.

There is a "list" option on the clone\_aomp.sh that provides useful information about each AOMP repository.
```
   $AOMP_REPOS/aomp/bin/clone_aomp.sh list
```
The above command will produce output like this showing you the location and branch of the repos in the AOMP\_REPOS directory and if there are any discrepencies with respect to the manifest file.<br>
<b>Development Branch:</b><br>
MANIFEST FILE: ~/git/aomp17.0/aomp/bin/../manifests/aomp_17.0.xml

<b>Release Branch:</b><br>
MANIFEST FILE: ~/git/aomp17.0/aomp/bin/../manifests/aomp_17.0-2.xml
```
  repo src       branch                 path                 repo name    last hash    updated           commitor         for author
  --------       ------                 ----                 ---------    ---------    -------           --------         ----------
 gerritgit amd-stg-open         llvm-project lightning/ec/llvm-project 11d5fa11d52c 2021-10-30      Ron Lieberman     Saiyedul Islam
       roc amd-stg-open rocm-compilersupport      ROCm-CompilerSupport ef72f4edb3dd 2021-10-12      Jeremy Newton      Jeremy Newton
       roc amd-stg-open     rocm-device-libs          ROCm-Device-Libs fc2eac750cb3 2021-10-26           pvellien           pvellien
  roctools     aomp-dev                flang                     flang 4f1282c59a76 2021-09-22             GitHub            ronlieb
  roctools     aomp-dev          aomp-extras               aomp-extras aefc0d6c7434 2021-10-18      Ron Lieberman      Ron Lieberman
  roctools     aomp-dev                 aomp                      aomp a94d9f43f3b0 2021-11-01        gregrodgers        gregrodgers
  roctools   rocm-5.4.x          rocprofiler               rocprofiler e140f47f3609 2021-10-08 Gerrit Code Review      Zhongyu Zhang
  roctools   rocm-5.4.x            roctracer                 roctracer d8ecefda4efd 2021-10-09      Ammar Elwazir      Ammar ELWazir
  roctools   rocm-5.4.x            ROCdbgapi                 ROCdbgapi 1040f1521831 2021-09-26 Laurent Morichetti Laurent Morichetti
  roctools   rocm-5.4.x               ROCgdb                    ROCgdb a1f2a479f060 2021-09-24 Laurent Morichetti       Simon Marchi
  roctools   rocm-5.4.x               hipamd                    hipamd bedc5f614221 2021-10-04  Pruthvi Madugundu            Sourabh
  roctools   rocm-5.4.x                  hip                       hip 3413a164f458 2021-10-11      Maneesh Gupta      Maneesh Gupta
  roctools   rocm-5.4.x               ROCclr                    ROCclr aba55f5c2775 2021-10-07      Zhongyu Zhang   German Andryeyev
       roc   rocm-5.4.x  ROCm-OpenCL-Runtime       ROCm-OpenCL-Runtime bf77cab71234 2021-09-27        Freddy Paul        Freddy Paul
       roc    roc-5.4.x             rocminfo                  rocminfo 1452f8fa24b2 2021-07-19      Icarus Sparry      Icarus Sparry
       roc rocm-rel-5.4           rocm-cmake                rocm-cmake 8d82398d269d 2021-09-14            Jenkins            Jenkins
       roc   rocm-5.4.x         rocr-runtime              ROCR-Runtime f32dfa887d02 2021-10-27         Sean Keely         Sean Keely
       roc    roc-5.4.x roct-thunk-interface      ROCT-Thunk-Interface 5b152ed0f043 2021-09-30         Sean Keely         Sean Keely
     rocsw rocm-rel-5.4              hipfort                   hipfort 96e9cb88281d 2021-11-16             GitHub   Dominic Charrier
```
For more information, or if you are interested in joining the development of AOMP, please read the AOMP developers README file located here [README](../bin/README.md).
