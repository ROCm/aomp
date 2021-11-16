# AOMP CentOS 8 Install
Currently, we support CentOS 8.1. If CentOS 7 is being used, the RHEL 7 rpm is recommended at this time.

AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.<br>

### Download and Install (CentOS 8)
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_14.0-0/aomp_CENTOS_8-14.0-0.x86_64.rpm
sudo rpm -i aomp_CENTOS_8-14.0-0.x86_64.rpm
```
### Download and Install (CentOS 7)
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_14.0-0/aomp_REDHAT_7-14.0-0.x86_64.rpm
sudo rpm -i aomp_REDHAT_7-14.0-0.x86_64.rpm
```
Confirm AOMP environment variable is set:
```
echo $AOMP
```

## Prerequisites
The ROCm kernel driver is required for AMD GPU support.
Also, to control access to the ROCm device, a user group "video" must be created and users need to be added to this group.

### AMD KFD Driver
<b>Install dkms tool</b>
```
  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`
```
Create a /etc/yum.repos.d/rocm.repo file with the following contents:
```
  [ROCm]
  name=ROCm
  baseurl=http://repo.radeon.com/rocm/centos8/rpm
  enabled=1
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
```
<b>Install rock-dkms</b>
```
  sudo yum install rock-dkms
```
### Set Group Access
```
  sudo reboot
  sudo usermod -a -G video $USER
```
### NVIDIA CUDA Driver
The CUDA installation is optional.
```
  wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-rhel8-10-2-local-10.2.89-440.33.01-1.0-1.x86_64.rpm
  sudo rpm -i cuda-repo-rhel8-10-2-local-10.2.89-440.33.01-1.0-1.x86_64.rpm
  sudo dnf clean all
  sudo dnf -y module install nvidia-driver:latest-dkms
  sudo dnf -y install cuda
```
