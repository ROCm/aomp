# AOMP Debian/Ubuntu Install 
AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.<br>

On Ubuntu 22.04,  run these commands:
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_20.0-1/aomp_Ubuntu2204_20.0-1_amd64.deb
sudo dpkg -i aomp_Ubuntu2204_20.0-1_amd64.deb
```
On Ubuntu 24.04,  run these commands:
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_20.0-1/aomp_Ubuntu2404_20.0-1_amd64.deb
sudo dpkg -i aomp_Ubuntu2404_20.0-1_amd64.deb
```

The AOMP bin directory (which includes the standard clang and llvm binaries) is not intended to be in your PATH for typical operation.

## Prerequisites
### AMD KFD Driver
These commands are for supported Debian-based systems and target only the amdgpu_dkms core component.
Install kernel headers:
```
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
```
```
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
```
Ubuntu 22.04:
```
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.3/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/amdgpu.list
```
Ubuntu 24.04:
```
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.3/ubuntu noble main" | sudo tee /etc/apt/sources.list.d/amdgpu.list
```

Update and Install:
```
sudo apt update
sudo apt install amdgpu-dkms
sudo reboot
sudo usermod -a -G render,video $USER
```

### NVIDIA CUDA Driver
The CUDA installation is optional.
Note these instructions reference the install for Ubuntu 22.04.
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```
Depending on your system the CUDA install could take a very long time.
