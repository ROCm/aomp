# Verify and Install Linux Support 
The ROCm kernel driver is required for AMD GPU support.
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
If you build AOMP with support for nvptx GPUs, you must first install CUDA 10.
Note these instructions reference the install for Ubuntu 16.04.

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
<b>Important Note:</b>
There is a conflict with the KFD when simultaneously running the GUI on SLES-15-SP1, which leads to unpredicatable behavior when offloading to the GPU. We recommend using SLES-15-SP1 in text mode to avoid running both the KFD and GUI at the same time.

SUSE SLES-15-SP1 comes with kfd support installed. To verify this:
```
  sudo dmesg | grep kfd
  sudo dmesg | grep amdgpu
```

### Set Group Access
```
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER
```

### NVIDIA CUDA Driver
If you build AOMP with support for nvptx GPUs, you must first install CUDA 10.

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

<b>Important Note:</b>
If using a GUI on SLES-15-SP1, such as gnome, the installation of CUDA may cause the GUI to fail to load. This seems to be caused by a symbolic link pointing to nvidia-libglx.so instead of xorg-libglx.so. This can be fixed by updating the symbolic link:
```
  sudo rm /etc/alternatives/libglx.so
  sudo ln -s /usr/lib64/xorg/modules/extensions/xorg/xorg-libglx.so /etc/alternatives/libglx.so
```
# CentOS/RedHat Support
Coming Soon.
