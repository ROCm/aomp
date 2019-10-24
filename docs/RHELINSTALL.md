# AOMP RHEL 7 Install 
AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.<br>
<br><b>The installation may need the following dependency:</b>
```
sudo yum install perl-Digest-MD5
```
### Download and Install
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_REDHAT_7-0.7-5.x86_64.rpm
sudo rpm -i aomp_REDHAT_7-0.7-5.x86_64.rpm
```
If CUDA is not installed the installation may cancel, to bypass this:
```
sudo rpm -i --nodeps aomp_REDHAT_7-0.7-5.x86_64.rpm
```
Confirm AOMP environment variable is set:
```
echo $AOMP
```

## Prerequisites
The ROCm kernel driver is required for AMD GPU support.
Also, to control access to the ROCm device, a user group "video" must be created and users need to be added to this group.

### AMD KFD Driver

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
### Set Group Access
```
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER
```
### NVIDIA CUDA Driver
To build AOMP with support for nvptx GPUs, you must first install CUDA 10.  We recommend CUDA 10.0.  CUDA 10.1 will not work until AOMP moves to the trunk development of LLVM 9. The CUDA installation is now optional.

<b>Download Instructions for CUDA (CentOS/RHEL 7)</b>
1. Go to https://developer.nvidia.com/cuda-10.0-download-archive
2. For SLES-15, select Linux®, x86_64, RHEL or CentOS, 7, rpm(local) and then click Download.
3. Navigate to the rpm in your Linux® directory and run the following commands:
```
  sudo rpm -i cuda-repo-rhel7-10-0-local-10.0.130-410.48-1.0-1.x86_64.rpm
  sudo yum clean all
  sudo yum install cuda
```
