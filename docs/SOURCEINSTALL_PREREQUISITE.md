## Prerequisites for Source Install of AOMP

### 1. Required Distribution Packages

#### Debian or Ubuntu Packages

```
   sudo apt-get install g++-5 g++-7 pkg-config libpci-dev libnuma-dev libffi-dev git python libopenmpi-dev gawk mesa-common-dev

   # Additional packages used by rocgdb
   sudo apt-get install python3 texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses5-dev liblzma-dev python3-setuptools python3-dev

```


#### SLES-15-SP1 Packages
```
  sudo zypper install -y git pciutils-devel python-base libffi-devel gcc gcc-c++ libnuma-devel patchutils openmpi2-devel mesa-libGL-devel libquadmath0

  A symbolic link may be required at /usr/lib64: /usr/lib64/libquadmath.so -> /usr/lib64/libquadmath.so.0.

  # Additional packages used by rocgdb
  sudo zypper install -y texinfo bison flex babeltrace-devel python3 python3-pip python3-devel python3-setuptools makeinfo ncurses-devel libexpat-devel xz-devel


```
#### RHEL 7 Packages
Building from source requires a newer gcc. Devtoolset-7 is recommended, follow instructions 1-3 here:<br>
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD.
https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/<br>

<b>The build_aomp.sh script will automatically enable devtoolset-7 if found in /opt/rh/devtoolset-7/enable. If you want to build an individual component you will need to manually start devtoolset-7 from the instructions above.</b><br>

```
  sudo yum install pciutils-devel numactl-devel libffi-devel mesa-libGL-devel

  # Additional packages used by rocgdb
  sudo yum install texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel
```
 RHEL 7.6 and earlier RHEL 7 versions do not have the python36-devel package, which requires a software collection installation.
```
  sudo subscription-manager repos --enable rhel-7-server-optional-rpms --enable rhel-server-rhscl-7-rpms
  sudo yum -y install rh-python36 rh-python36-python-tools
  scl enable rh-python36 bash
```

RHEL 7.7 and later RHEL 7 versions
```
  sudo yum install python3 python3-pip python36-devel python36-setuptools
```
#### CentOS 8 Packages
```
  sudo yum install dnf-plugins-core
  sudo yum config-manager --set-enabled powertools

  sudo yum install gcc gcc-c++ git make pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libquadmath-devel python3 python3-pip python36-devel python3-setuptools python2

  # Additional packages used by rocgdb
  sudo yum install texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel
  python3 -m pip install CppHeaderParser argparse wheel lit --user
```

### 2. User-installed Python Components

After all the required system package from section 1 are installed, there are some python packages that must be locally installed by the user building AOMP. Use this command to install these.  Do not install these as root.

```
  python3 -m pip install CppHeaderParser argparse wheel lit
```

### 3.  Build CMake 3.13.4 in /usr/local/cmake

We have seen problems with newer versions of cmake. We have only verified version 3.13.4 for the various component builds necessary for aomp. All invocations of cmake in the build scripts use $AOMP_CMAKE.  The default for the AOMP_CMAKE variable is /usr/local/cmake/bin/cmake.  Use these commands to install cmake 3.13.4 from source into /usr/local/cmake.

```
  $ sudo apt-get install libssl-dev
  $ mkdir /tmp/cmake
  $ cd /tmp/cmake
  $ wget https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4.tar.gz
  $ tar -xvzf cmake-3.13.4.tar.gz
  $ cd cmake-3.13.4
  $ ./bootstrap --prefix=/usr/local/cmake
  $ make
  $ sudo make install
```
Alternatively, you could change the --prefix option to install cmake 3.13.4 somewhere else. Then be sure to change the value of he environment variable AOMP_CMAKE to be the cmake binary.

### 4. Verify KFD Driver

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

### 5. Create the Unix Video Group
```
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER
```

### 6. Optional Install CUDA

The Nvidia CUDA SDK is NOT required to build AOMP or install the AOMP package. 
However, to build AOMP from source, you SHOULD have the Nvidia CUDA SDK version 10 installed because AOMP may be used to build applications for NVIDIA GPUs. The current default build list of Nvidia subarchs is "30,35,50,60,61,70".  For example, the default list will support application builds with --offload-arch=sm_30 and --offload-arch=sm_60 etc.  This build list can be changed with the NVPTXGPUS environment variable as shown above.


### 7. Optional Install of Spack

If you expect to install AOMP sources using the release source tarball with spack, you must install Spack. Refer to [these install instructions](https://spack.readthedocs.io/en/latest/getting_started.html#installation) for instructions on installing spack.
The AOMP spack configuration file is currently missing proper dependencies, so be sure to install the packages listed above before proceeding with source install via spack.
