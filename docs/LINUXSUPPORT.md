# Verify and Install Linux Support 
The ROCm kernel driver is required for AMD GPU support and CUDA is required for nvptx GPU support.
Also, to control access to the ROCm device, a linux user group "video" must be created and users added to this group.

# Debian or Ubuntu Support
### AMD KFD Driver
These commands are for supported Debian-based systems and target only the rock_dkms core component. More information can be found [HERE](https://rocm.github.io/ROCmInstall.html#ubuntu-support---installing-from-a-debian-repository).
```
echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rock-dkms

sudo reboot
sudo usermod -a -G video $USER
```
### NVIDIA CUDA Driver
To build AOMP with support for nvptx GPUs, you must first install CUDA 10.  We recommend CUDA 10.0.  CUDA 10.1 will not work until AOMP moves to the trunk development of LLVM 9. The CUDA installation is now optional. Note these instructions reference the install for Ubuntu 16.04.

<b>Download Instructions for CUDA (Ubuntu 16.04)</b>
1. Go to https://developer.nvidia.com/cuda-10.0-download-archive
2. For Ubuntu 16.04, select Linux, x86_64, Ubuntu, 16.04, deb(local) and then click Download. Note you can change these options for your specific distribution type.
3. Navigate to the debian in your Linux directory and run the following commands:
```
   sudo dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
   sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
   sudo apt-get update
   sudo apt-get install cuda
```
Depending on your system the CUDA install could take a very long time.

# SUSE SLES-15-SP1 Support
### KFD for AMD GPUs
SUSE SLES-15-SP1 comes with kfd support installed. To verify this:
```
  sudo dmesg | grep kfd
  sudo dmesg | grep amdgpu
```

### Set Group Access
```
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo usermod -a -G video $USER
```

### NVIDIA CUDA Driver
To build AOMP with support for nvptx GPUs, you must first install CUDA 10.  We recommend CUDA 10.0.  CUDA 10.1 will not work until AOMP moves to the trunk development of LLVM 9. The CUDA installation is now optional.

<b>Download Instructions for CUDA (SLES15)</b>
1. Go to https://developer.nvidia.com/cuda-10.0-download-archive
2. For SLES-15, select Linux, x86_64, SLES, 15.0, rpm(local) and then click Download.
3. Navigate to the rpm in your Linux directory and run the following commands:
```
  sudo rpm -i cuda-repo-sles15-10-0-local-10.0.130-410.48-1.0-1.x86_64.rpm
  sudo zypper refresh
  sudo zypper install cuda
```
If prompted, select the 'always trust key' option. Depending on your system the CUDA install could take a very long time.

# CentOS/RedHat Support
Coming Soon.
