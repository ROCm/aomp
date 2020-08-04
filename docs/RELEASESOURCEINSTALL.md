# Build and Install From Release Source Tarball

The AOMP build and install from the release source tarball can be done manually or with spack.
Building from source requires a number of platform dependencies.
These dependencies are not yet provided with the spack configuration file.
So if you are building from source either manually or building with spack, you must install the prerequisites for the platforms listed below.

## Source Build Prerequisites

To build AOMP from source you must: 1. install certain distribution packages, 2. ensure the KFD kernel module is installed and operating, 3. create the Unix video group, and 4. install spack if required.

### 1. Required Distribution Packages

#### Debian or Ubuntu Packages

```
   sudo apt-get install cmake g++-5 g++ pkg-config libpci-dev libnuma-dev libelf-dev libffi-dev git python libopenmpi-dev gawk

   # Additional packages used by rocgdb
   sudo apt-get install texinfo libbison-dev bison flex libbabeltrace-dev python-pip libncurses5-dev liblzma-dev
   python -m pip install CppHeaderParser argparse

```
#### SLES-15-SP1 Packages

```
  sudo zypper install -y git pciutils-devel cmake python-base libffi-devel gcc gcc-c++ libnuma-devel libelf-devel patchutils openmpi2-devel

  # Additional packages used by rocgdb
  SUSEConnect --product sle-module-python2/15.1/x86_64
  sudo zypper install -y texinfo bison flex babeltrace-devel python-pip python-devel makeinfo ncurses-devel libexpat-devel xz-devel

  python -m pip install wheel CppHeaderParser argparse

```
#### RHEL 7  Packages
Building from source requires a newer gcc. Devtoolset-7 is recommended, follow instructions 1-3 here:<br>
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD.
https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/<br>

<b>The build_aomp.sh script will automatically enable devtoolset-7 if found in /opt/rh/devtoolset-7/enable. If you want to build an individual component you will need to manually start devtoolset-7 from the instructions above.</b><br>

```
  sudo yum install cmake3 pciutils-devel numactl-devel libffi-devel

  # Additional packages used by rocgdb
  sudo yum install texinfo bison flex python-pip python-devel ncurses-devel.x86_64 expat-devel.x86_64 xz-devel.x86_64 libbabeltrace-devel.x86_64

  python -m pip install wheel CppHeaderParser argparse

```
The build scripts use cmake, so we need to link cmake --> cmake3 in /usr/bin
```
  sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
```

### 2. Verify KFD Driver

Please verify you have the proper software installed as AOMP needs certain support to function properly, such as the KFD driver for AMD GPUs.

#### Debian or Ubuntu Support
These commands are for supported Debian-based systems and target only the rock_dkms core component. More information can be found [HERE](https://rocm.github.io/ROCmInstall.html#ubuntu-support---installing-from-a-debian-repository).
```
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rock-dkms
```

#### SUSE SLES-15-SP1 Support
<b>Important Note:</b>
There is a conflict with the KFD when simultaneously running the GUI on SLES-15-SP1, which leads to unpredicatable behavior when offloading to the GPU. We recommend using SLES-15-SP1 in text mode to avoid running both the KFD and GUI at the same time.

SUSE SLES-15-SP1 comes with kfd support installed. To verify this:
```
  sudo dmesg | grep kfd
  sudo dmesg | grep amdgpu
```

#### RHEL 7 Support
```
  sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms
  sudo subscription-manager repos --enable rhel-7-server-optional-rpms
  sudo subscription-manager repos --enable rhel-7-server-extras-rpms
  sudo rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
```
<b>Install dkms tool</b>
```
  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`
```
Create a /etc/yum.repos.d/rocm.repo file with the following contents:
```
  [ROCm]
  name=ROCm
  baseurl=http://repo.radeon.com/rocm/yum/rpm
  enabled=1
  gpgcheck=0
```
<b>Install rock-dkms</b>
```
  sudo yum install rock-dkms
```
### 3. Create the Unix Video Group
Regardless of Linux distribution, you must create a video group to contain the users authorized to use the GPU. 
```
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER
```
### 4. Install spack
To use spack to build and install from the release source tarball, you must install spack first.
Please refer to
[these install instructions](https://spack.readthedocs.io/en/latest/getting_started.html#installation) for instructions on installing spack.
Remember,the aomp spack configuration file is currently missing dependencies, so be sure to install the packages listed above before proceeding.

## Build AOMP manually from release source tarball

To build and install aomp from the release source tarball run these commands:

```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_11.7-1/aomp-11.7-1.tar.gz
   tar -xzf aomp-11.7-1.tar.gz
   cd aomp
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
The build_aomp.sh script will install into $HOME/rocm/aomp_11.7-1 and create a symbolic link from $HOME/rocm/aomp to $HOME/rocm/aomp_11.7-1.
This feature allows multiple versions of AOMP to be installed concurrently.
To enable a backlevel version of AOMP, simply set AOMP to $HOME/rocm/aomp_11.7-0.

## Build AOMP with spack

Assuming your have installed the prerequisites listed above, use these commands to fetch the source and build aomp. 
Currently the aomp configuration is not yet in the spack git hub so you must create the spack package first. 

```
   wget https://github.com/ROCm-Developer-Tools/aomp/blob/master/bin/package.py
   spack create -n aomp -t makefile --force https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_11.7-1/aomp-11.7-1.tar.gz
   spack edit aomp
   spack install aomp
```
The "spack create" command will download and start an editor of a newly created spack config file.
With the exception of the sha256 value, copy the contents of the downloaded package.py file into
into the spack configuration file. You may restart this editor with the command "spack edit aomp"

Depending on your system, the "spack install aomp" command could take a very long time.
Unless you set the AOMP environment variable, AOMP will be installed in /usr/local/aomp_<RELEASE> with a symbolic link from /usr/local/aomp to /usr/local/aomp_<RELEASE>.
Be sure you have write access to /usr/local or set AOMP to a location where you have write access.
