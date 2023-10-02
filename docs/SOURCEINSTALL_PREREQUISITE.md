## Prerequisites for Source Install of AOMP

### 1. Required Distribution Packages

#### Debian or Ubuntu Packages

```
   sudo apt-get install gcc g++ pkg-config libpci-dev libnuma-dev libffi-dev git python libopenmpi-dev gawk mesa-common-dev libtool libdrm-amdgpu1 libdrm-dev ccache libdw-dev libgtest-dev libsystemd-dev

   # ubuntu 22 distributions seem to be missing libstdc++12
   sudo apt-get install  libstdc++-12-dev
   
   # Additional packages used by rocgdb
   sudo apt-get install python3 texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses5-dev liblzma-dev python3-setuptools python3-dev libpython3.8-dev libudev-dev libgmp-dev

```


#### SLES-15-SP4 Packages
```
  sudo zypper install -y git pciutils-devel python-base libffi-devel gcc gcc-c++ libnuma-devel patchutils openmpi2-devel mesa-libGL-devel libquadmath0 libtool libdrm libdrm-devel ccache gcc-gfortran libdw-devel libgtest-devel systemd-devel libX-devel

  A symbolic link may be required at /usr/lib64: /usr/lib64/libquadmath.so -> /usr/lib64/libquadmath.so.0.

  # Additional packages used by rocgdb
  sudo zypper install -y texinfo bison flex babeltrace-devel python3 python3-pip python3-devel python3-setuptools makeinfo ncurses-devel libexpat-devel xz-devel libgmp-devel


```
#### RHEL 7 Packages
Building from source requires a newer gcc. Devtoolset-7 is recommended, follow instructions 1-3 here:<br>
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD.
https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/<br>

<b>The build_aomp.sh script will automatically enable devtoolset-7 if found in /opt/rh/devtoolset-7/enable. If you want to build an individual component you will need to manually start devtoolset-7 from the instructions above.</b><br>

```
  sudo yum install pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool libdrm libdrm-devel ccache gcc-gfortran libdw-devel libgtest-devel systemd-devel

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

  sudo yum install gcc gcc-c++ git make pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libquadmath-devel python3 python3-pip python36-devel python3-setuptools python2 libtool libdrm libdrm-devel ccache gcc-gfortran libdw-devel libgtest-devel systemd-devel

  # Additional packages used by rocgdb
  sudo yum install texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel gmp-devel

  # To build aomp with Ninja set AOMP_USE_NINJA=1 . You need this installed with dnf
  dnf install ninja-build
```

#### CentOS 9 Packages
```
  sudo yum install dnf-plugins-core gcc gcc-c++ git make pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libquadmath-devel python3 python3-pip python3-devel python3-setuptools libtool libdrm libdrm-devel ccache gcc-gfortran libdw-devel libgtest-devel systemd-devel

  # Additional packages used by rocgdb
  sudo yum install texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel gmp-devel
```

### 2. User-installed Python Components

After all the required system package from section 1 are installed, there are some python packages that must be locally installed by the user building AOMP. Use this command to install these.  Do not install these as root.

```
  python3 -m pip install CppHeaderParser argparse wheel lit lxml barectf termcolor
```

### 3.  Build CMake in /usr/local/cmake

This can also be done with ./build_prereq.sh, which installs to $HOME/local/cmake.<br>
All invocations of cmake in the build scripts use $AOMP_CMAKE.  The default for the AOMP_CMAKE variable is /usr/local/cmake/bin/cmake. You can also use these commands to install cmake 3.22.1 manually from source into /usr/local/cmake instead of using ./build_prereq.sh.

```
  $ sudo apt-get install libssl-dev
  $ mkdir /tmp/cmake
  $ cd /tmp/cmake
  $ wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1.tar.gz
  $ tar -xvzf cmake-3.22.1.tar.gz
  $ cd cmake-3.22.1
  $ ./bootstrap --prefix=/usr/local/cmake
  $ make
  $ sudo make install
```
Alternatively, you could change the --prefix option to install cmake somewhere else. Then be sure to change the value of he environment variable AOMP_CMAKE to be the cmake binary.

### 4. Verify KFD Driver

Please verify you have the proper software installed as AOMP needs certain support to function properly, such as the KFD driver for AMD GPUs.
More information can be found [HERE](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

#### Debian or Ubuntu Support
These commands are for supported Debian-based systems and target only the amdgpu_dkms core component.
```
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
```
Ubuntu 20.04:
```
echo 'deb [arch=amd64] http://repo.radeon.com/amdgpu/latest/ubuntu focal main' | sudo tee /etc/apt/sources.list.d/amdgpu.list
```
Update and Install:
```
sudo apt update
sudo apt install amdgpu-dkms
```

#### SUSE SLES-15-SP4 Support
<b>Important Note:</b>
There is a conflict with the KFD when simultaneously running the GUI on SLES-15-SP4, which leads to unpredicatable behavior when offloading to the GPU. We recommend using SLES-15-SP4 in text mode to avoid running both the KFD and GUI at the same time.

SUSE SLES-15-SP4 comes with kfd support installed. To verify this:
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
Install kernel headers:
```
  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`
```
Create a /etc/yum.repos.d/amdgpu.repo file with the following contents:
```
  [amdgpu]
  name=amdgpu
  baseurl=https://repo.radeon.com/amdgpu/latest/rhel/7.9/main/x86_64
  enabled=1
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
```
Install amdgpu-dkms:
```
  sudo yum install amdgpu-dkms
```

### 5. Create the Unix Video Group
```
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER
```

### 6. Optional Install CUDA

The Nvidia CUDA SDK is NOT required to build AOMP or install the AOMP package. 
However, to build AOMP from source, you SHOULD have the Nvidia CUDA SDK version 10/11 installed because AOMP may be used to build applications for NVIDIA GPUs. The current default build list of Nvidia subarchs is "30,35,50,60,61,70".  For example, the default list will support application builds with --offload-arch=sm_30 and --offload-arch=sm_60 etc.  This build list can be changed with the NVPTXGPUS environment variable as shown above.


### 7. Optional Install of Spack

If you expect to install AOMP sources using the release source tarball with spack, you must install Spack. Refer to [these install instructions](https://spack.readthedocs.io/en/latest/getting_started.html#installation) for instructions on installing spack.
The AOMP spack configuration file is currently missing proper dependencies, so be sure to install the packages listed above before proceeding with source install via spack.
