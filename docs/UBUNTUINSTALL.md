# AOMP Debian/Ubuntu Install 
AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.<br>

On Ubuntu 20.04,  run these commands:
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_13.0-6/aomp_Ubuntu2004_13.0-6_amd64.deb
sudo dpkg -i aomp_Ubuntu2004_13.0-6_amd64.deb
```

On Ubuntu 18.04 LTS (bionic beaver), run these commands:
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_13.0-6/aomp_Ubuntu1804_13.0-6_amd64.deb
sudo dpkg -i aomp_Ubuntu1804_13.0-6_amd64.deb
```

The AOMP bin directory (which includes the standard clang and llvm binaries) is not intended to be in your PATH for typical operation.

## Prerequisites
### AMD KFD Driver
These commands are for supported Debian-based systems and target only the rock_dkms core component. More information can be found [HERE](https://rocm.github.io/ROCmInstall.html#ubuntu-support---installing-from-a-debian-repository).
```
sudo apt install wget gnupg2
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rock-dkms

sudo reboot
sudo usermod -a -G video $USER
```
### NVIDIA CUDA Driver
The CUDA installation is optional.
Note these instructions reference the install for Ubuntu 18.04.
```
   wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64
   sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
   sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
   sudo apt-get update
   sudo apt-get install cuda
```
Depending on your system the CUDA install could take a very long time.
