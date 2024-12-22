# AOMP RHEL 8/9 Install
Currently, we support RHEL 8.10/9.4.

AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.<br>

### Download and Install (RHEL 9)
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_20.0-1/aomp_REDHAT_9-20.0-1.x86_64.rpm
sudo rpm -i aomp_REDHAT_9-20.0-1.x86_64.rpm
```
### Download and Install (RHEL 8)
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_20.0-1/aomp_REDHAT_8-20.0-1.x86_64.rpm
sudo rpm -i aomp_REDHAT_8-20.0-1.x86_64.rpm
```

## Prerequisites
The ROCm kernel driver is required for AMD GPU support.
Also, to control access to the ROCm device, a user group "video" must be created and users need to be added to this group.

RHEL 9
```
  sudo dnf install "kernel-headers-$(uname -r)" "kernel-devel-$(uname -r)" "kernel-devel-matched-$(uname -r)"
```
RHEL 8
```
  sudo dnf install "kernel-headers-$(uname -r)" "kernel-devel-$(uname -r)"
```

### AMD KFD Driver

RHEL 9
```
sudo tee /etc/yum.repos.d/amdgpu.repo <<EOF
[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/6.3/el/9.4/main/x86_64/
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

sudo dnf clean all
```

RHEL 8
```
sudo tee /etc/yum.repos.d/amdgpu.repo <<EOF
[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/6.3/el/8.10/main/x86_64/
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

sudo dnf clean all
```

```
sudo dnf install amdgpu-dkms
sudo reboot
```

```
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
