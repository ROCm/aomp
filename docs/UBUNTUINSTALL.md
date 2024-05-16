# AOMP Debian/Ubuntu Install 
AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.<br>

On Ubuntu 20.04,  run these commands:
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_19.0-0/aomp_Ubuntu2004_19.0-0_amd64.deb
sudo dpkg -i aomp_Ubuntu2004_19.0-0_amd64.deb
```
On Ubuntu 22.04,  run these commands:
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_19.0-0/aomp_Ubuntu2204_19.0-0_amd64.deb
sudo dpkg -i aomp_Ubuntu2204_19.0-0_amd64.deb
```

The AOMP bin directory (which includes the standard clang and llvm binaries) is not intended to be in your PATH for typical operation.

## Prerequisites
### AMD KFD Driver
These commands are for supported Debian-based systems and target only the amdgpu_dkms core component.
```
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
```
Ubuntu 20.04:
```
echo 'deb [arch=amd64] http://repo.radeon.com/amdgpu/latest/ubuntu focal main' | sudo tee /etc/apt/sources.list.d/amdgpu.list
```
Update and Install:
```
sudo apt update
sudo apt install amdgpu-dkms
sudo reboot
sudo usermod -a -G render video $USER
```
### NVIDIA CUDA Driver
The CUDA installation is optional.
Note these instructions reference the install for Ubuntu 20.04.
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
Depending on your system the CUDA install could take a very long time.
