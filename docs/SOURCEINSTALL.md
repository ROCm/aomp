# Source Install V 13.1-0

Build and install from sources is possible.  However, the source build for AOMP is complex for several reasons.
- Many repos are required.  The clone_aomp.sh script ensures you have all repos and the correct branch.
- Requires that Cuda SDK 10 is installed for NVIDIA GPUs. ROCm does not need to be installed for AOMP.
- It is a bootstrapped build. The built and installed LLVM compiler is used to build library components.
- Additional package dependencies are required that are not required when installing the AOMP package.

## Source Build Prerequisites

To build AOMP from source you must: 1. Install certain distribution packages, 2. Build CMake 3.13.4 from source 3. ensure the KFD kernel module is installed and operating, 4. create the Unix video group, and 5. install spack if required. [This link](SOURCEINSTALL_PREREQUISITE.md) provides instructions to satisfy all the AOMP source build dependencies.

## Clone and Build AOMP

<b>Choose a Build Version (Development or Release)</b>
The development version is the next version to be released.  It is possible that the development version is broken due to regressions that often occur during development.

Starting with AOMP 13.1, AOMP uses the repo command instead of a long sequence of git clone commands. A new script called init\_aomp\_repos.sh will do the initial fetch of the repositories.

Here are the commands to do a source build of AOMP_14.0.

```
   cd /tmp
   export AOMP_VERSION=14.0
   export AOMP_REPOS=$HOME/git/aomp${AOMP_VERSION}
   wget https://github.com/ROCm-Developer-Tools/aomp/raw/aomp-dev/init_aomp_repos.sh
   chmod 755 init_aomp_repos.sh
   ./init_aomp_repos.sh
   cd $AOMP_REPOS/aomp/bin
   ./clone_aomp.sh
   nohup ./build_aomp.sh &
```

Change the value of AOMP\_REPOS to the directory name where you want to store all the repositories needed for AOMP. It is recommended that name include the AOMP_VERSION.
The init_aomp_repos.sh, clone_aomp.sh, and build_aomp.sh are expected to take a long time to execute.

<b>To build a previous release of AOMP:</b>
```
   cd $AOMP_REPOS/aomp/bin
   export AOMP_VERSION=13.1
   export AOMP_REPOS=$HOME/git/aomp${AOMP_VERSION}
   git checkout aomp-13.1-0
   git pull
   export AOMP_CHECK_GIT_BRANCH=0 //Tags will be used to checkout various repos. This will ignore the detached head state to avoid build errors.
   ./clone_aomp.sh
   nohup ./build_aomp.sh &
```
For more information, or if you are interested in joining the development of AOMP, please read the AOMP developers README file located here [README](../bin/README.md).
