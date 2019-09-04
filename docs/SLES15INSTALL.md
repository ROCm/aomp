# AOMP SUSE SLES-15-SP1 Install 
AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-1/aomp_SLES15_SP1-0.7-1.x86_64.rpm
sudo rpm -i aomp_SLES15_SP1-0.7-1.x86_64.rpm
```
Confirm AOMP environment variable is set:
```
echo $AOMP
```
