# AOMP RPM Install 
<!--
### RPM Install
For rpm-based Linux distributions, use this rpm
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/r/aomp-0.6-3.x86_64.rpm
sudo rpm -i aomp-0.6-3.x86_64.rpm
```
-->
### No root RPM install

By default, the packages install their content to the release directory /opt/rocm/aomp_0.X-Y and then a  symbolic link is created at /opt/rocm/aomp to the release directory. This requires root access.

To install the rpm package without root access into your home directory, you can run these commands.
```
   mkdir /tmp/temproot ; cd /tmp/temproot 
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.6-3/aomp-0.6-3.x86_64.rpm
   rpm2cpio aomp-0.6-3.x86_64.rpm | cpio -idmv
   mv /tmp/temproot/opt/rocm $HOME
   export PATH=$PATH:$HOME/rocm/aomp/bin
   export AOMP=$HOME/rocm/aomp
```
The last two commands could be put into your .bash_profile file so you can always access the compiler.

