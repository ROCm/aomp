# Install Without Root
By default, the packages install their content to the release directory /usr/lib/aomp_0.X-Y and then a  symbolic link is created at /usr/lib/aomp to the release directory. This requires root access.

Once installed go to [TESTINSTALL](TESTINSTALL.md) for instructions on getting started with AOMP examples.

### Debian
To install the debian package without root access into your home directory, you can run these commands.<br>
On Ubuntu 18.04 LTS (bionic beaver):
```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_11.0-1/aomp_Ubuntu1804_11.0-1_amd64.deb
   dpkg -x aomp_Ubuntu1804_11.0-1_amd64.deb /tmp/temproot
```
On Ubuntu 16.04:
```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_11.0-1/aomp_Ubuntu1604_11.0-1_amd64.deb
   dpkg -x aomp_Ubuntu1604_11.0-1_amd64.deb /tmp/temproot
```
```
   mv /tmp/temproot/usr $HOME
   export PATH=$PATH:$HOME/usr/lib/aomp/bin
   export AOMP=$HOME/usr/lib/aomp
```
The last two commands could be put into your .bash_profile file so you can always access the compiler.

### RPM
To install the rpm package without root access into your home directory, you can run these commands.
```
   mkdir /tmp/temproot ; cd /tmp/temproot 
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_11.0-1/aomp_REDHAT_7-11.0-1.x86_64.rpm
   rpm2cpio aomp_REDHAT_7-11.0-1.x86_64.rpm | cpio -idmv
   mv /tmp/temproot/usr/lib $HOME
   export PATH=$PATH:$HOME/rocm/aomp/bin
   export AOMP=$HOME/rocm/aomp
```
The last two commands could be put into your .bash_profile file so you can always access the compiler.
