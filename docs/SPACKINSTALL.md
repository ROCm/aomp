# Spack Source Install V 0.7-5 (DEV)

Build and install from sources is possible with spack.  Source build requires build dependencies.  These dependencies are not yet provided with the spack configuration file.  So if you are using spack to build aomp, you should install the correct OS dependencies before you spack build. 

## Building AOMP from source requires these dependencies:<br>
<b>Ubuntu</b>

```
   sudo apt-get install cmake g++-5 g++ pkg-config libpci-dev libnuma-dev libelf-dev libffi-dev git python libopenmpi-dev gawk
```
<b>SLES-15-SP1</b>

```
  sudo zypper install -y git pciutils-devel cmake python-base libffi-devel gcc gcc-c++ libnuma-devel libelf-devel patchutils openmpi2-devel
```
<b>RHEL 7</b><br>
Building from source requires a newer gcc. Devtoolset-7 is recommended, follow instructions 1-3 here:<br>
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD.
https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/<br>

```
  sudo yum install cmake3 pciutils-devel numactl-devel libffi-devel
```
The build scripts use cmake, so we need to link cmake --> cmake3 in /usr/bin
```
  sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
```

## Verify KFD Driver

Please verify you have the proper software installed as AOMP needs certain support to function properly, such as the KFD driver for AMD GPUs.

### Debian or Ubuntu Support
These commands are for supported Debian-based systems and target only the rock_dkms core component. More information can be found [HERE](https://rocm.github.io/ROCmInstall.html#ubuntu-support---installing-from-a-debian-repository).
```
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rock-dkms
```

### SUSE SLES-15-SP1 Support
<b>Important Note:</b>
There is a conflict with the KFD when simultaneously running the GUI on SLES-15-SP1, which leads to unpredicatable behavior when offloading to the GPU. We recommend using SLES-15-SP1 in text mode to avoid running both the KFD and GUI at the same time.

SUSE SLES-15-SP1 comes with kfd support installed. To verify this:
```
  sudo dmesg | grep kfd
  sudo dmesg | grep amdgpu
```

### RHEL 7 Support
```
  sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms
  sudo subscription-manager repos --enable rhel-7-server-optional-rpms
  sudo subscription-manager repos --enable rhel-7-server-extras-rpms
  sudo rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
```
<b>Install and setup Devtoolset-7</b></br>
Devtoolset-7 is recommended, follow instructions 1-3 here:<br>
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD.
https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/<br>

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

## Set Group Access
```
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER
```

## Build AOMP from released source with spack

Currently the aomp configuration is not yet in the spack git hub. 
Assuming you have already installed spack, use these commands to fetch the source and build. 
These command will only work after a release of aomp. 

```
   spack create -n aomp -t makefile --force https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-3/aomp-0.7-4.tar.gz
   spack install aomp
```

Depending on your system, these  commands could take a very long time.
