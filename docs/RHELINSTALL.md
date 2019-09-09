# AOMP RHEL 7 Install 
AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.<br>
<br><b>The installation may need the following dependency:</b>
```
sudo yum install perl-Digest-MD5
```
### Download and Install
```
wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-2/aomp_REDHAT_7-0.7-2.x86_64.rpm
sudo rpm -i aomp_REDHAT_7-0.7-2.x86_64.rpm
```
If CUDA is not installed the installation may cancel, to bypass this:
```
sudo rpm -i --nodeps aomp_REDHAT_7-0.7-2.x86_64.rpm
```
Confirm AOMP environment variable is set:
```
echo $AOMP
```
