# AOMP SUSE SLES-15-SP4 Install 
AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_20.0-0/aomp_SLES15_SP4-20.0-0.x86_64.rpm
sudo rpm -i aomp_SLES15_SP4-20.0-0.x86_64.rpm
```
Confirm AOMP environment variable is set:
```
echo $AOMP
```

## Prerequisites
The ROCm kernel driver is required for AMD GPU support.
Also, to control access to the ROCm device, a user group "video" must be created and users need to be added to this group.

### AMD KFD DRIVER
<b>Important Note:</b>
There is a conflict with the KFD when simultaneously running the GUI on SLES-15-SP4, which leads to unpredicatable behavior when offloading to the GPU. We recommend using SLES-15-SP4 in text mode to avoid running both the KFD and GUI at the same time.

SUSE SLES-15-SP4 comes with kfd support installed. To verify this:
```
  sudo dmesg | grep kfd
  sudo dmesg | grep amdgpu
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

<b>Important Note:</b>
If using a GUI on SLES-15-SP4, such as gnome, the installation of CUDA may cause the GUI to fail to load. This seems to be caused by a symbolic link pointing to nvidia-libglx.so instead of xorg-libglx.so. This can be fixed by updating the symbolic link:
```
  sudo rm /etc/alternatives/libglx.so
  sudo ln -s /usr/lib64/xorg/modules/extensions/xorg/xorg-libglx.so /etc/alternatives/libglx.so
```
