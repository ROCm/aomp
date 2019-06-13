Summary: AMD OpenMP Compiler Suite
Name: aomp_SLES15_SP1
Version: __VERSION1
Release: __VERSION3_MOD 
Source: ~/rpm/SOURCES/aomp_SLES15_SP1.tar.gz
URL: https://github.com/ROCm-Developer-Tools/aomp
License: none
Group: System/Base
Vendor: AMD

%description
The AMD OpenMP Compiler (AOMP). 

%prep
%setup -n %{name}

%build

%install
mkdir -p $RPM_BUILD_ROOT/usr/lib
rsync -a usr/lib $RPM_BUILD_ROOT/usr

%clean
rm -rf $RPM_BUILD_ROOT
echo rm -rf %{_tmppath}/%{name}
rm -rf %{_tmppath}/%{name}
echo rm -rf %{_topdir}/BUILD/%{name}
rm -rf %{_topdir}/BUILD/%{name}

%post
if [ -L /usr/lib/aomp ] ; then rm /usr/lib/aomp ; fi
ln -sf /usr/lib/aomp___VERSION2_STRING /usr/lib/aomp
if [ -L /usr/bin/aompversion ] ; then rm /usr/bin/aompversion ; fi
ln -sf /usr/lib/aomp/bin/aompversion /usr/bin/aompversion 
if [ -L /usr/bin/mymcpu ] ; then rm /usr/bin/mymcpu ; fi
ln -sf /usr/lib/aomp/bin/mymcpu /usr/bin/mymcpu
if [ -L /usr/bin/mygpu ] ; then rm /usr/bin/mygpu ; fi
ln -sf /usr/lib/aomp/bin/mygpu /usr/bin/mygpu
if [ -L /usr/bin/cloc.sh ] ; then rm /usr/bin/cloc.sh ; fi
ln -sf /usr/lib/aomp/bin/cloc.sh /usr/bin/cloc.sh

%files
%defattr(-,root,root)
%{_prefix}/lib/aomp___VERSION2_STRING/*

%changelog
* Thu Jun 13 2019 Greg Rodgers <gregory.rodgers@amd.com>
