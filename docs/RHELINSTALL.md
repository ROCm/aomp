# AOMP RHEL 7 Install
Currently, we support RHEL 7.4 and RHEL 7.6.

AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.<br>
<br><b>The installation may need the following dependency:</b>
```
sudo yum install perl-Digest-MD5 perl-URI-Encode
```
### Download and Install
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_19.0-2/aomp_CENTOS_7-19.0-2.x86_64.rpm
sudo rpm -i aomp_CENTOS_7-19.0-2.x86_64.rpm
```
If CUDA is not installed the installation may cancel, to bypass this:
```
sudo rpm -i --nodeps aomp_CENTOS_7-19.0-2.x86_64.rpm
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
Create a /etc/yum.repos.d/amdgpu.repo file with the following contents:
```
  [amdgpu]
  name=amdgpu
  baseurl=https://repo.radeon.com/amdgpu/latest/rhel/7.9/main/x86_64
  enabled=1
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
```
<b>Install amdgpu-dkms</b>
```
  sudo yum install amdgpu-dkms
```
### Set Group Access
```
  sudo reboot
  sudo usermod -a -G video $USER
```
### NVIDIA CUDA Driver
The CUDA installation is optional.
```
  wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-rhel7-10-0-local-10.0.130-410.48-1.0-1.x86_64
  sudo rpm -i cuda-repo-rhel7-10-0-local-10.0.130-410.48-1.0-1.x86_64.rpm
  sudo yum clean all
  sudo yum install cuda
```
