## Prerequisites for Source Install of AOMP

### 1. Required Distribution Packages

#### Debian or Ubuntu Packages

```
   sudo apt-get install cmake g++-5 g++-7 pkg-config libpci-dev libnuma-dev libelf-dev libffi-dev git python libopenmpi-dev gawk

   # Additional packages used by rocgdb
   sudo apt-get install python3 texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses5-dev liblzma-dev python3-setuptools python3-dev
   python3 -m pip install CppHeaderParser argparse wheel

```

#### SLES-15-SP1 Packages
```
  sudo zypper install -y git pciutils-devel cmake python-base libffi-devel gcc gcc-c++ libnuma-devel libelf-devel patchutils openmpi2-devel

  # Additional packages used by rocgdb
  sudo zypper install -y texinfo bison flex babeltrace-devel python3 python3-pip python3-devel python3-setuptools makeinfo ncurses-devel libexpat-devel xz-devel

  python3 -m pip install CppHeaderParser argparse wheel

```
#### RHEL 7  Packages
Building from source requires a newer gcc. Devtoolset-7 is recommended, follow instructions 1-3 here:<br>
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD.
https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/<br>

<b>The build_aomp.sh script will automatically enable devtoolset-7 if found in /opt/rh/devtoolset-7/enable. If you want to build an individual component you will need to manually start devtoolset-7 from the instructions above.</b><br>

```
  sudo yum install cmake3 pciutils-devel numactl-devel libffi-devel

  # Additional packages used by rocgdb
  sudo yum install texinfo bison flex ncurses-devel.x86_64 expat-devel.x86_64 xz-devel.x86_64 libbabeltrace-devel.x86_64
```
 RHEL 7.6 and earlier RHEL 7 versions do not have the python36-devel package, which requires a software collection installation.
```
  sudo subscription-manager repos --enable rhel-7-server-optional-rpms --enable rhel-server-rhscl-7-rpms
  sudo yum -y install rh-python36 rh-python36-python-tools
  scl enable rh-python36 bash
  python3 -m pip install CppHeaderParser argparse wheel
```

RHEL 7.7 and later RHEL 7 versions
```
  sudo yum install python3 python3-pip python36-devel python36-setuptools
  python3 -m pip install CppHeaderParser argparse wheel
```

The build scripts use cmake, so we need to link cmake --> cmake3 in /usr/bin
```
  sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
```
2.  Build CMake 3.13.4 in /usr/local/cmake
The default for the AOMP_CMAKE variable is /usr/local/cmake/bin/cmake.
Use these commands to install cmake 3.13.4 from source into /usr/local/cmake.

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

### 3. Verify KFD Driver

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

### 4. Create the Unix Video Group
```
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER
```

### 5. Optional Install of Spack

If you expect to install AOMP sources using the release source tarball with spack, you must install Spack. Refer to [these install instructions](https://spack.readthedocs.io/en/latest/getting_started.html#installation) for instructions on installing spack.
The AOMP spack configuration file is currently missing proper dependencies, so be sure to install the packages listed above before proceeding with source install via spack.
