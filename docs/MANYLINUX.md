# Install From Release Binary Tarball

AOMP releases are now built on AlmaLinux (manylinux) can be installed on various operating systems from the release binary tarball. We currently support AlmaLinux 8, SLES15, RHEL 8, RHEl 9, Ubuntu 22.04, and Ubuntu 24.04.

```
   cd /usr/local
   wget https://github.com/ROCm/aomp/releases/download/rel_23.0-0/aomp-23.0-0.tar.gz
   tar -xzf aomp-23.0-0.tar.gz
   ln -s aomp_23.0-0 aomp
```
```
AOMP = /usr/local/aomp
```
