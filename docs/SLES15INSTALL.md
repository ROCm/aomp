# AOMP SUSE SLES-15-SP5 Install 
AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_20.0-1/aomp_SLES15_SP5-20.0-1.x86_64.rpm
sudo rpm -i aomp_SLES15_SP5-20.0-1.x86_64.rpm
```

## Prerequisites
The ROCm kernel driver is required for AMD GPU support.
Also, to control access to the ROCm device, a user group "video" must be created and users need to be added to this group.

### AMD KFD DRIVER
Install kernel headers:
```
sudo zypper install kernel-default-devel
```

```
sudo tee /etc/zypp/repos.d/amdgpu.repo <<EOF
[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/6.3/sle/15.5/main/x86_64/
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF
```
```
sudo zypper ref
sudo zypper --gpg-auto-import-keys install amdgpu-dkms
sudo reboot
```

### Set Group Access
```
  sudo usermod -a -G video $USER
```

### NVIDIA CUDA Driver
The CUDA installation is optional.
```
  wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-sles15-10-0-local-10.0.130-410.48-1.0-1.x86_64
  sudo rpm -i cuda-repo-sles15-10-0-local-10.0.130-410.48-1.0-1.x86_64.rpm
  sudo zypper refresh
  sudo zypper install cuda
```
If prompted, select the 'always trust key' option. Depending on your system the CUDA install could take a very long time.
