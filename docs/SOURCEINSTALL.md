# Source Install V 13.0-5

Build and install from sources is possible.  However, the source build for AOMP is complex for several reasons.
- Many repos are required.  The clone_aomp.sh script ensures you have all repos and the correct branch.
- Requires that Cuda SDK 10 is installed for NVIDIA GPUs. ROCm does not need to be installed for AOMP.
- It is a bootstrapped build. The built and installed LLVM compiler is used to build library components.
- Additional package dependencies are required that are not required when installing the AOMP package.

## Source Build Prerequisites

To build AOMP from source you must: 1. Install certain distribution packages, 2. Build CMake 3.13.4 from source 3. ensure the KFD kernel module is installed and operating, 4. create the Unix video group, and 5. install spack if required. [This link](SOURCEINSTALL_PREREQUISITE.md) provides instructions to satisfy all the AOMP source build dependencies.

## Clone and Build AOMP

Starting with AOMP 13.1, AOMP uses the repo command instead of a long sequence of git clone commands.
One cannot use git clone to fetch aomp because this is not compatible with how repo command works.
So a new script called init\_aomp\_repos.sh will do the initial fetch of the repositories.
Here are the commands to do a source build of AOMP_13.1-x.
```
   cd /tmp
   export AOMP_VERSION=13.1
   export AOMP_REPOS=$HOME/git/aomp${AOMP_VERSION}
   wget https://github.com/ROCm-Developer-Tools/aomp/raw/aomp-dev/init_aomp_repos.sh
   chmod 755 init_aomp_repos.sh
   ./init_aomp_repos.sh
   $AOMP_REPOS/aomp/bin/clone_aomp.sh
   nohup $AOMP_REPOS/aomp/bin/build_aomp.sh &
```

Depending on your system, the last three commands could take a very long time. For more information, please refer to the AOMP developers README file located [HERE](../bin/README.md).

You only need to do the checkout/pull in the AOMP repository. The file "bin/aomp_common_vars" lists the branches of each repository for a particular AOMP release. In the master branch of AOMP, aomp_common_vars lists the development branches. It is a good idea to run clone_aomp.sh twice after you checkout a release to be sure you pulled all the checkouts for a particular release.

If you are interested in joining the development of AOMP, please read the details on the source build at [README](../bin/README.md).
