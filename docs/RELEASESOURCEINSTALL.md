# Build and Install From Release Source Tarball

The AOMP build and install from the release source tarball can be done manually.
Building from source requires a number of platform dependencies.

## Source Build Prerequisites

To build AOMP from source you must: 1. Install certain distribution packages, 2. Build CMake 3.25.2 from source, this can be done with build_prereq.sh 3. ensure the KFD kernel module is installed and operating, and 4. create the Unix video group. [This link](SOURCEINSTALL_PREREQUISITE.md) provides instructions to satisfy all the AOMP source build dependencies.

## Build AOMP manually from release source tarball

To build and install aomp from the release source tarball run these commands:

```
   wget https://github.com/ROCm/aomp/releases/download/rel_23.0-0/aomp-23.0-0-source.tar.gz
   tar -xzf aomp-23.0-0-source.tar.gz
   cd aomp23.0
   nohup make &
```
Depending on your system, the last command could take a very long time.  So it is recommended to use nohup and background the process.  The simple Makefile that make will use runs build script "build_aomp.sh" and sets some flags to avoid git checks and applying ROCm patches. Here is that Makefile:
```
AOMP ?= /usr/local/aomp
AOMP_REPOS = $(shell pwd)
all:
        AOMP=$(AOMP) AOMP_REPOS=$(AOMP_REPOS) AOMP_CHECK_GIT_BRANCH=0 AOMP_APPLY_ROCM_PATCHES=0 $(AOMP_REPOS)/aomp/bin/build_aomp.sh
```
If you set the environment variable AOMP, the Makefile will install to that directory.
Otherwise, the Makefile will install into /usr/local.
So you must have authorization to write into /usr/local if you do not set the environment variable AOMP.
Let's assume you set the environment variable AOMP to "$HOME/rocm/aomp" in .bash_profile.
The build_aomp.sh script will install into $HOME/rocm/aomp_23.0-0 and create a symbolic link from $HOME/rocm/aomp to $HOME/rocm/aomp_23.0-0.
This feature allows multiple versions of AOMP to be installed concurrently.
To enable a backlevel version of AOMP, simply set AOMP to $HOME/rocm/aomp_22.0-2.
