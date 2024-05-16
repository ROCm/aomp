#!/bin/bash
# 
#  check_amdgpu_modversion.sh:  Check amdgpu kernel module version
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
[ -f $thisdir/aomp_common_vars ] && . $thisdir/aomp_common_vars
# --- end standard header ----

which dpkg 2>/dev/null >/dev/null
if [ $? == 0 ] ; then
_packaged_amdgpu_ver=`dpkg -l amdgpu-dkms 2>/dev/null | tail -n 1 | cut -d":" -f2 | cut -d"." -f1-3`
else
_packaged_amdgpu_ver=""
fi
if [ "$_packaged_amdgpu_ver" == "" ] ; then
   ROCM_EXPECTED_MODVERSION=${ROCM_EXPECTED_MODVERSION:-6.3.6}
else
   ROCM_EXPECTED_MODVERSION=$_packaged_amdgpu_ver
fi
_llvm_install_dir=${AOMP:-/opt/rocm/llvm}
which modinfo >/dev/null 2>/dev/null
if [ $? == 0 ] ; then
   _amdgpu_mod_version=`modinfo -F version amdgpu`
   if [ ! -z $_amdgpu_mod_version ] ; then
     if [ "$_amdgpu_mod_version" != "$ROCM_EXPECTED_MODVERSION" ] ; then
        if [ -f  $_llvm_install_dir/bin/aompversion ] ; then
           _aomp_version_string=`$AOMP/bin/aompversion`
           _phrase="for AOMP version $_aomp_version_string"
        else
           _phrase="for $_llvm_install_dir"
        fi
        echo
        echo "WARNING: Unexpected version of amdgpu kernel module found on this system: $_amdgpu_mod_version"
        echo "         The expected version $_phrase is $ROCM_EXPECTED_MODVERSION"
        echo "         Execution of compiled binaries may have issues on this system. For best results"
        echo "         consider installing the latest "amdgpu-dkms" package from ROCm and reboot."
        echo "         Command to check amdgpu kernel module version:  modinfo -F version amdgpu"
        echo
	exit 0  # returning non zero fails the run_rocm_test.sh
     fi
   fi
fi
exit 0
