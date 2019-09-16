# AOMP Debian/Ubuntu Install 

On Ubuntu 18.04 LTS (bionic beaver), run these commands:
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-3/aomp_Ubuntu1804_0.7-3_amd64.deb
sudo dpkg -i aomp_Ubuntu1804_0.7-3_amd64.deb
```
On Ubuntu 16.04,  run these commands:
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-3/aomp_Ubuntu1604_0.7-3_amd64.deb
sudo dpkg -i aomp_Ubuntu1604_0.7-3_amd64.deb
```
The AOMP bin directory (which includes the standard clang and llvm binaries) is not intended to be in your PATH for typical operation.

### No root Debian Install

By default, the packages install their content to the release directory /usr/lib/aomp_0.X-Y and then a  symbolic link is created at /usr/lib/aomp to the release directory. This requires root access.

To install the debian package without root access into your home directory, you can run these commands.<br>
On Ubuntu 18.04 LTS (bionic beaver):
```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-3/aomp_Ubuntu1804_0.7-3_amd64.deb
   dpkg -x aomp_Ubuntu1804_0.7-3_amd64.deb /tmp/temproot
```
On Ubuntu 16.04:
```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-3/aomp_Ubuntu1604_0.7-3_amd64.deb
   dpkg -x aomp_Ubuntu1604_0.7-3_amd64.deb /tmp/temproot
```
```
   mv /tmp/temproot/usr $HOME
   export PATH=$PATH:$HOME/usr/lib/aomp/bin
   export AOMP=$HOME/usr/lib/aomp
```
The last two commands could be put into your .bash_profile file so you can always access the compiler.
