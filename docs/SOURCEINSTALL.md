# Source Install V 13.0-6

Build and install from sources is possible.  However, the source build for AOMP is complex for several reasons.
- Many repos are required.  The clone_aomp.sh script ensures you have all repos and the correct branch.
- Requires that Cuda SDK 10 is installed for NVIDIA GPUs. ROCm does not need to be installed for AOMP.
- It is a bootstrapped build. The built and installed LLVM compiler is used to build library components.
- Additional package dependencies are required that are not required when installing the AOMP package.

## Source Build Prerequisites

To build AOMP from source you must: 1. Install certain distribution packages, 2. Build CMake 3.13.4 from source 3. ensure the KFD kernel module is installed and operating, 4. create the Unix video group, and 5. install spack if required. [This link](SOURCEINSTALL_PREREQUISITE.md) provides instructions to satisfy all the AOMP source build dependencies.

## Clone and Build AOMP
```
   cd $HOME ; mkdir -p git/aomp13 ; cd git/aomp13
   git clone https://github.com/rocm-developer-tools/aomp
   cd $HOME/git/aomp13/aomp/bin
```

<b>Choose a Build Version (Development or Release)</b>
The development version is the next version to be released.  It is possible that the development version is broken due to regressions that often occur during development.

<b>For the Development Branch :</b>
```
   git checkout aomp-dev
   git pull
```
Starting with AOMP 13.1, AOMP uses the repo command instead of a long sequence of git clone commands.  One cannot use git clone to fetch aomp because this is not compatible with how repo command works.  So a new script called init\_aomp\_repos.sh will do the initial fetch of the repositories.

Here are the commands to do a source build of AOMP_13.1-x.

```
   cd /tmp
   export AOMP_VERSION=13.1
   export AOMP_REPOS=$HOME/git/aomp${AOMP_VERSION}
   wget https://github.com/ROCm-Developer-Tools/aomp/raw/aomp-dev/init_aomp_repos.sh
   ./init_aomp_repos.sh
   nohup $AOMP_REPOS/aomp/bin/build_aomp.sh &
```

If instead, you want to build from the sources of a previous release such as 13.0-6 that is possible as well.

<b>For the Release Branch:</b>
```
   git checkout aomp-13.0-6
   git pull
   export AOMP_CHECK_GIT_BRANCH=0 //Tags will be used to checkout various repos. This will ignore the detached head state to avoid build errors.
```
<b>Clone and Build:</b>
```
   ./clone_aomp.sh
   ./build_aomp.sh
```

Depending on your system, the last two commands could take a very long time. For more information, please refer to the AOMP developers README file located [HERE](../bin/README.md).

You only need to do the checkout/pull in the AOMP repository. The file "bin/aomp_common_vars" lists the branches of each repository for a particular AOMP release. In the master branch of AOMP, aomp_common_vars lists the development branches. It is a good idea to run clone_aomp.sh twice after you checkout a release to be sure you pulled all the checkouts for a particular release.

If you are interested in joining the development of AOMP, please read the details on the source build at [README](../bin/README.md).
