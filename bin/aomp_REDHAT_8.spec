Summary: AMD OpenMP Compiler Suite
Name: aomp_REDHAT_8
Version: __VERSION1
Release: __VERSION3_MOD
Source: ~/rpm/SOURCES/aomp_REDHAT_8.tar.gz
URL: https://github.com/ROCm-Developer-Tools/aomp
License: MIT and ASL 2.0
Group: System/Base
Vendor: AMD
AutoReq: no

%define debug_package %{nil}
%define __os_install_post %{nil}
%define __requires_exclude (^perl)|(^lib(amdhip|hip).*$)|(^libcuda\\.so\\..*$)|(^libhsa.*)|(^librocm_smi.*)

%description
 The AMD OpenMP Compiler (AOMP) is an experimental LLVM compiler
 suite for offloading to either Radeon GPUs or Nvidia GPUs.
 AOMP requires the dkms module from ROCm, amdgpu-dkms.

%prep
%setup -n %{name}

%build

%install
echo "INSTALL RUNNING IN $PWD"
mkdir -p $RPM_BUILD_ROOT/usr/lib
rsync -a usr/lib $RPM_BUILD_ROOT/usr

%clean
echo "CLEAN RUNNING IN $PWD"
rm -rf $RPM_BUILD_ROOT
echo rm -rf %{_tmppath}/%{name}
rm -rf %{_tmppath}/%{name}
echo rm -rf %{_topdir}/BUILD/%{name}
rm -rf %{_topdir}/BUILD/%{name}

%post
echo "POST INSTALL SCRIPT FROM spec file RUNNING IN $PWD"
if [ -L /usr/lib/aomp ] ; then rm /usr/lib/aomp ; fi
ln -sf /usr/lib/aomp___VERSION2_STRING /usr/lib/aomp
if [ -L /usr/bin/aompversion ] ; then rm /usr/bin/aompversion ; fi
ln -sf /usr/lib/aomp/llvm/bin/aompversion /usr/bin/aompversion
if [ -L /usr/bin/gpurun ] ; then rm /usr/bin/gpurun ; fi
ln -sf /usr/lib/aomp/llvm/bin/gpurun /usr/bin/gpurun
echo "DONE POST INSTALL SCRIPT FROM spec file RUNNING IN $PWD"

%files
%defattr(-,root,root)
%{_prefix}/lib/aomp___VERSION2_STRING

%postun
rm /usr/lib/aomp
rm /usr/bin/aompversion
rm /usr/bin/gpurun

%changelog
* Thu Aug 2 2019 Greg Rodgers <gregory.rodgers@amd.com>
