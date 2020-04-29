# Source Install V 11.5-1 (DEV)

Build and install from sources is possible.  However, the source build for AOMP is complex for several reasons.
- Many repos are required.  The clone_aomp.sh script ensures you have all repos and the correct branch.
- Requires that Cuda SDK 10 is installed for NVIDIA GPUs. ROCm does not need to be installed for AOMP.
- It is a bootstrapped build. The built and installed LLVM compiler is used to build library components.
- Additional package dependencies are required that are not required when installing the AOMP package.

## Source Build Prerequisites

### 1. Required Distribution Packages

#### Debian or Ubuntu Packages

```
   sudo apt-get install cmake g++-5 g++ pkg-config libpci-dev libnuma-dev libelf-dev libffi-dev git python libopenmpi-dev gawk mesa-common-dev
```
#### SLES-15-SP1 Packages

```
  sudo zypper install -y git pciutils-devel cmake python-base libffi-devel gcc gcc-c++ libnuma-devel libelf-devel patchutils openmpi2-devel
```
#### RHEL 7  Packages
Building from source requires a newer gcc. Devtoolset-7 is recommended, follow instructions 1-3 here:<br>
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD.
https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/<br>

<b>The build_aomp.sh script will automatically enable devtoolset-7 if found in /opt/rh/devtoolset-7/enable. If you want to build an individual component you will need to manually start devtoolset-7 from the instructions above.</b><br>

```
  sudo yum install cmake3 pciutils-devel numactl-devel libffi-devel
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
```
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER
```

## Clone and Build AOMP

```
   cd $HOME ; mkdir -p git/aomp ; cd git/aomp
   git clone https://github.com/rocm-developer-tools/aomp
   cd $HOME/git/aomp/aomp/bin
```

<b>Choose a Build Version (Development or Release)</b>
The development version is the next version to be released.  It is possible that the development version is broken due to regressions that often occur during development.  If instead, you want to build from the sources of a previous release such as 11.5-0 that is possible as well.

<b>For the Development Branch:</b>
```
   git checkout amd-stg-openmp
   git pull
```

<b>For the Release Branch:</b>
```
   git checkout rel_11.5-0
   git pull
```
<b>Clone and Build:</b>
```
   ./clone_aomp.sh
   ./build_aomp.sh
```
Depending on your system, the last two commands could take a very long time. For more information, please refer to the AOMP developers README file located [HERE](../bin/README.md).

You only need to do the checkout/pull in the AOMP repository. The file "bin/aomp_common_vars" lists the branches of each repository for a particular AOMP release. In the master branch of AOMP, aomp_common_vars lists the development branches. It is a good idea to run clone_aomp.sh twice after you checkout a release to be sure you pulled all the checkouts for a particular release.

If you are interested in joining the development of AOMP, please read the details on the source build at [README](../bin/README.md).
