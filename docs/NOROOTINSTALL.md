# Install Without Root
By default, the packages install their content to the release directory /usr/lib/aomp_0.X-Y and then a  symbolic link is created at /usr/lib/aomp to the release directory. This requires root access.

Once installed go to [TESTINSTALL](TESTINSTALL.md) for instructions on getting started with AOMP examples.

### Debian
To install the debian package without root access into your home directory, you can run these commands.<br>

On Ubuntu 22.04:
```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_20.0-2/aomp_Ubuntu2204_20.0-2_amd64.deb
   dpkg -x aomp_Ubuntu2204_20.0-2_amd64.deb /tmp/temproot

   Also can be done with aomp-hip-libraries_Ubuntu2204_20.0-2_amd64.deb
```
On Ubuntu 24.04:
```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_20.0-2/aomp_Ubuntu2404_20.0-2_amd64.deb
   dpkg -x aomp_Ubuntu2404_20.0-2_amd64.deb /tmp/temproot

   Also can be done with aomp-hip-libraries_Ubuntu2404_20.0-2_amd64.deb
```
Move to $HOME and set variables:
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
```
For SLES15-SP5:
```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_20.0-2/aomp_SLES15_SP5-20.0-2.x86_64.rpm
   rpm2cpio aomp_SLES15_SP5-20.0-2.x86_64.rpm | cpio -idmv

   Also can be done with aomp-hip-libraries_SLES15_SP5-20.0-2.x86_64.rpm
```
For RHEL 8:
```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_20.0-2/aomp_REDHAT_8-20.0-2.x86_64.rpm
   rpm2cpio aomp_REDHAT_8-20.0-2.x86_64.rpm | cpio -idmv

   Also can be done with aomp-hip-libraries_REDHAT_8-20.0-2.x86_64.rpm
```
For RHEL 9:
```
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_20.0-2/aomp_REDHAT_9-20.0-2.x86_64.rpm
   rpm2cpio aomp_REDHAT_9-20.0-2.x86_64.rpm | cpio -idmv

   Also can be done with aomp-hip-libraries_REDHAT_9-20.0-2.x86_64.rpm
```
Move to $HOME and set variables:
```
   mv /tmp/temproot/usr $HOME
   export PATH=$PATH:$HOME/usr/lib/aomp/bin
   export AOMP=$HOME/usr/lib/aomp
```
The last two commands could be put into your .bash_profile file so you can always access the compiler.
