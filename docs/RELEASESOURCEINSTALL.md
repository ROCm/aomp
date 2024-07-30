# Build and Install From Release Source Tarball

The AOMP build and install from the release source tarball can be done manually or with spack.
Building from source requires a number of platform dependencies.
These dependencies are not yet provided with the spack configuration file.
So if you are building from source either manually or building with spack, you must install the prerequisites for the platforms listed below.

## Source Build Prerequisites

To build AOMP from source you must: 1. Install certain distribution packages, 2. Build CMake 3.22.1 from source, this can be done with build_prereq.sh 3. ensure the KFD kernel module is installed and operating, 4. create the Unix video group, and 5. install spack if required. [This link](SOURCEINSTALL_PREREQUISITE.md) provides instructions to satisfy all the AOMP source build dependencies.

## Build AOMP manually from release source tarball

To build and install aomp from the release source tarball run these commands:

```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_19.0-3/aomp-19.0-3.tar.gz
   tar -xzf aomp-19.0-3.tar.gz
   cd aomp19.0
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
The build_aomp.sh script will install into $HOME/rocm/aomp_19.0-3 and create a symbolic link from $HOME/rocm/aomp to $HOME/rocm/aomp_19.0-3.
This feature allows multiple versions of AOMP to be installed concurrently.
To enable a backlevel version of AOMP, simply set AOMP to $HOME/rocm/aomp_19.0-2.

## Build AOMP with spack

Assuming your have installed the [prerequisites](SOURCEINSTALL_PREREQUISITE.md), use these commands to fetch the source and build aomp. Currently the aomp configuration is not yet in the spack git hub so you must create the spack package first.

```
   wget https://github.com/ROCm-Developer-Tools/aomp/blob/aomp-19.0-3/bin/package.py
   spack create -n aomp -t makefile --force https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_19.0-3/aomp-19.0-3.tar.gz
   spack edit aomp
   spack install aomp
```
The "spack create" command will download and start an editor of a newly created spack config file.
With the exception of the sha256 value, copy the contents of the downloaded package.py file into
into the spack configuration file. You may restart this editor with the command "spack edit aomp"

Depending on your system, the "spack install aomp" command could take a very long time.
Unless you set the AOMP environment variable, AOMP will be installed in /usr/local/aomp_<RELEASE> with a symbolic link from /usr/local/aomp to /usr/local/aomp_<RELEASE>.
Be sure you have write access to /usr/local or set AOMP to a location where you have write access.
