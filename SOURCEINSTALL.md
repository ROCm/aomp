### Source Install
Build and install from sources is possible.  However, the source build for AOMP is complex for several reasons.
- Many repos are required.  The clone_aomp.sh script ensures you have all repos and the correct branch.
- Requires that Cuda SDK 10 is installed. ROCm does not need to be installed for AOMP.
- It is a bootstrapped build. The built and installed LLVM compiler is used to build library components.
- Additional package dependencies are required that are not required when installing the AOMP package.

Building AOMP from source requires these dependencies that can be resolved on an ubuntu system with apt-get.

```
   sudo apt-get install cmake g++-5 g++ pkg-config libpci-dev libnuma-dev libelf-dev libffi-dev git python libopenmpi-dev
```
The ROCm kernel driver is required for AMD GPU support. These commands are for supported Debian-based systems and target only the rock_dkms core component. More information can be found [HERE](https://rocm.github.io/ROCmInstall.html#ubuntu-support---installing-from-a-debian-repository).
```
echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo install rock_dkms

sudo reboot
sudo usermod -a -G video $LOGNAME
```
To build AOMP with support for nvptx GPUs, you must first install CUDA 10.  We recommend CUDA 10.0.  CUDA 10.1 will not work till AOMP moves to the trunk development of LLVM 9.  Once you download CUDA 10.0 local install file, these commands should complete the install of CUDA. The CUDA installation is now optional. Note the first command references the install for Ubuntu 16.04.
```
   sudo dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
   sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
   sudo apt-get update
   sudo apt-get install cuda
```
To build AOMP from source, run these commands.
```
   cd $HOME ; mkdir -p git/aomp ; cd git/aomp
   git clone https://github.com/rocm-developer-tools/aomp
   cd $HOME/git/aomp/aomp/bin
   git checkout master
   ./clone_aomp.sh
   ./build_aomp.sh
```
Depending on your system, the last two commands and the CUDA install could take a very long time. For more information, please refer to the AOMP developers README file located [HERE](bin/README.md).

The source build process above builds the development version of AOMP by checking out the master branch of AOMP.   The development version is the next version to be released.  It is possible that the development version is broken due to regressions that often occur during development.  If you want to build from the sources of a previous release such as 0.6-0, run these commands before running clone_aomp.sh.
```
   git checkout rel_0.6.0
   git pull
```
You only need to do this in the AOMP repository. The file "bin/aomp_common_vars" lists the branches of each repository for a particular AOMP release. In the master branch of AOMP, aomp_common_vars lists the development branches. It is a good idea to run clone_aomp.sh twice after you checkout a release to be sure you pulled all the checkouts for a particular release.

If your are interested in joining the development of AOMP, please read the details on the source build at [README](bin/README.md).


