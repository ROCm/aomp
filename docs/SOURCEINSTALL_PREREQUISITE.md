## Prerequisites for Source Install of AOMP

### 1. Required Distribution Packages

#### Debian or Ubuntu Packages

```
  sudo apt-get install wget gcc g++ pkg-config libpci-dev libnuma-dev libffi-dev git python3 libopenmpi-dev gawk mesa-common-dev libtool libdrm-amdgpu1 libdrm-dev ccache libdw-dev libgtest-dev libsystemd-dev cmake openssl libssl-dev libgmp-dev libmpfr-dev libelf-dev pciutils python3.10-dev libudev-dev libgtest-dev libstdc++-12-dev python3-lxml ocl-icd-opencl-dev libsystemd-dev

  # Additional packages used by rocgdb
  sudo apt-get install texinfo libbison-dev bison flex libbabeltrace-dev python3-pip libncurses5-dev liblzma-dev python3-setuptools python3-dev libudev-dev libgmp-dev libmpfr-dev libdw-dev
```

Ubuntu 22.04 Only
```   
  python3.10-dev
```

Ubuntu 24.04 Only
```
  python3-barectf python3-pip python3-pip-whl python3-requests python3-venv python3-yaml
```

#### SLES-15-SP5 Packages
```
  sudo zypper install wget libopenssl-devel elfutils libelf-devel git pciutils-devel libffi-devel gcc gcc-c++ libnuma-devel openmpi2-devel Mesa-libGL-devel libquadmath0 libtool libX11-devel systemd-devel hwdata unzip mpfr-devel ocl-icd-devel

  A symbolic link may be required at /usr/lib64: /usr/lib64/libquadmath.so -> /usr/lib64/libquadmath.so.0.

  # Additional packages used by rocgdb and rocprofiler
  sudo zypper install texinfo bison flex babeltrace-devel python3 python3-pip python3-devel python3-setuptools makeinfo ncurses-devel libexpat-devel xz-devel libgmp-devel libatomic libdwarf-devel gtest-devel libdw-devel
```

#### RHEL Packages
RHEL 8
```
  sudo yum install dnf-plugins-core gcc-c++ git wget openssl-devel elfutils-libelf-devel elfutils-devel ccache pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool libdrm libdrm-devel gmp-devel rpm-build gcc-gfortran libdw-devel libgtest-devel systemd-devel mpfr-devel python38 python38-devel ocl-icd-devel libatomic libquadmath-devel

  # Additional packages used by rocgdb and roctracer
  sudo yum install texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel libatomic libdwarf-devel gtest-devel
```

RHEL 9
```
  sudo dnf install dnf-plugins-core gcc-c++ git wget openssl-devel elfutils-libelf-devel elfutils-devel ccache pciutils-devel numactl-devel libffi-devel mesa-libGL-devel libtool libdrm libdrm-devel gmp-devel rpm-build gcc-gfortran libdw-devel libgtest-devel systemd-devel mpfr-devel python3-devel ocl-icd-devel libatomic libquadmath-devel

  # Additional packages used by rocgdb and roctracer
  sudo dnf install texinfo bison flex ncurses-devel expat-devel xz-devel libbabeltrace-devel libatomic libdwarf-devel gtest-devel
  
  # To build aomp with Ninja set AOMP_USE_NINJA=1 . You need this installed with dnf
  dnf install ninja-build
```

### 2. User-installed Python Components

After all the required system package from section 1 are installed, there are some python packages that must be locally installed by the user building AOMP. Use this command to install these.  Do not install these as root.  

Ubuntu 22.04
```
  python3 -m pip install --ignore-installed --no-cache-dir barectf==3.1.2 PyYAML==5.3.1; python3 -m pip install CppHeaderParser argparse wheel lit lxml pandas
```

Ubuntu 24.04
```
  python3 -m venv /opt/venv; PATH=/opt/venv/bin:$PATH python3 -m pip install CppHeaderParser argparse lxml pandas setuptools PyYAML pandas
```

RHEL 8/9 and SLES15
```
  python3 -m pip install CppHeaderParser argparse wheel lit lxml barectf termcolor pandas
```

### 3. cmake

This section is no longer required.

The requirement for a specific version of cmake is now satisfied with the build_prereq.sh script
which is called by build_aomp.sh. The distribution cmake installed above is only required for
the first execution of build_prereq.sh. The AOMP build scripts are found in the bin directory of the aomp repository and are described in the developwer README.

### 4. Verify KFD Driver
#### IMPORTANT: We strongly recommend installing the amdgpu-dkms from 6.3 due to bug fixes.  
Please verify you have the proper software installed as AOMP needs certain support to function properly, such as the KFD driver (amdgpu-dkms) for AMD GPUs.
More information can be found [HERE](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

#### Debian or Ubuntu Support
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
```

#### SUSE SLES-15-SP5 Support
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

#### RHEL 8 Support
Install kernel headers:
```
  sudo dnf install "kernel-headers-$(uname -r)" "kernel-devel-$(uname -r)"
```
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
```

Install amdgpu-dkms:
```
  sudo dnf clean all
  sudo dnf install amdgpu-dkms
  sudo reboot
```

#### RHEL 9 Support
Install kernel headers:
```
  sudo dnf install "kernel-headers-$(uname -r)" "kernel-devel-$(uname -r)" "kernel-devel-matched-$(uname -r)"
```
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
```

Install amdgpu-dkms:
```
  sudo dnf clean all
  sudo dnf install amdgpu-dkms
  sudo reboot
```

### 5. Create the Unix Video Group
Ubuntu
```
  sudo usermod -a -G video,render $USER
```

All Other Operating Systems
```
  sudo usermod -a -G video $USER
```

### 6. Optional Install CUDA

The Nvidia CUDA SDK is NOT required to build AOMP or install the AOMP package. 
However, to build AOMP from source, you SHOULD have the Nvidia CUDA SDK version 10/11 installed because AOMP may be used to build applications for NVIDIA GPUs. The current default build list of Nvidia subarchs is "30,35,50,60,61,70".  For example, the default list will support application builds with --offload-arch=sm_30 and --offload-arch=sm_60 etc.  This build list can be changed with the NVPTXGPUS environment variable as shown above.

See [these install instructions](https://developer.nvidia.com/cuda-toolkit-archive)


### 7. Optional Install of Spack

If you expect to install AOMP sources using the release source tarball with spack, you must install Spack. Refer to [these install instructions](https://spack.readthedocs.io/en/latest/getting_started.html#installation) for instructions on installing spack.
The AOMP spack configuration file is currently missing proper dependencies, so be sure to install the packages listed above before proceeding with source install via spack.
