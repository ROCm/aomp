# AOMP CentOS 7/8 Install
Currently, we support CentOS 7/8.1/9.

AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.<br>

### Download and Install (CentOS 9)
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_19.0-2/aomp_CENTOS_9-19.0-2.x86_64.rpm
sudo rpm -i aomp_CENTOS_9-19.0-2.x86_64.rpm
```
### Download and Install (CentOS 8)
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_19.0-2/aomp_CENTOS_8-19.0-2.x86_64.rpm
sudo rpm -i aomp_CENTOS_8-19.0-2.x86_64.rpm
```
### Download and Install (CentOS 7)
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_19.0-2/aomp_CENTOS_7-19.0-2.x86_64.rpm
sudo rpm -i aomp_CENTOS_7-19.0-2.x86_64.rpm
```
Confirm AOMP environment variable is set:
```
echo $AOMP
```

## Prerequisites
The ROCm kernel driver is required for AMD GPU support.
Also, to control access to the ROCm device, a user group "video" must be created and users need to be added to this group.

### AMD KFD Driver
Install kernel headers:
```
  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`
```
Create a /etc/yum.repos.d/amdgpu.repo file with the following contents:
```
  [amdgpu]
  name=amdgpu
  baseurl=https://repo.radeon.com/amdgpu/latest/rhel/9.1/main/x86_64
  enabled=1
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
```
For CentOS 8 use:
```
  baseurl=https://repo.radeon.com/amdgpu/latest/rhel/8.4/main/x86_64
```
For CentOS 7 use:
```
  baseurl=https://repo.radeon.com/amdgpu/latest/rhel/7.9/main/x86_64
```

Install amdgpu-dkms:
```
  sudo yum install amdgpu-dkms
```
Set group access:
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
