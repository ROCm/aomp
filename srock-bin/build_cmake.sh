#!/bin/bash
#
# build_cmake.sh : build cmake and ninja
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
SUPPLEMENTAL_COMPONENTS=${SUPPLEMENTAL_COMPONENTS:-openmpi silo hdf5 fftw ninja}
PREREQUISITE_COMPONENTS=${PREREQUISITE_COMPONENTS:-cmake rocmsmilib hwloc aqlprofile rocm-core}

# --- Start standard header to set SROCK environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
# shellcheck disable=SC1091
. "$thisdir/srock_common_vars"
# --- end standard header ----
FLANG=${FLANG:-flang}

function runcmd(){
   THISCMD=$1
   if [ "$DRYRUN" ] ; then
      echo "$THISCMD"
   else
      echo "$THISCMD" 
      echo "$THISCMD" >>"$CMDLOGFILE"
      $THISCMD
      rc=$?
      if [ $rc != 0 ] ; then
         echo "ERROR:  The following command failed with return code $rc: "
         echo "        $THISCMD"
         exit $rc
      fi
   fi
}

function runcmdout(){
   THISCMD=$1
   OUTFILE=$2
   if [ "$DRYRUN" ] ; then
      echo "$THISCMD > $OUTFILE"
   else
      echo "$THISCMD > $OUTFILE"
      echo "$THISCMD > $OUTFILE" >>"$CMDLOGFILE"
      $THISCMD > "$OUTFILE"
      rc=$?
      if [ $rc != 0 ] ; then
         echo "ERROR:  The following command failed with return code $rc: "
         echo "        $THISCMD > $OUTFILE"
         exit $rc
      fi
   fi
}

function runcmdin(){
   THISCMD=$1
   INFILE=$2
   if [ "$DRYRUN" ] ; then
      echo "$THISCMD < $INFILE"
   else
      echo "$THISCMD < $INFILE"
      echo "$THISCMD < $INFILE" >>"$CMDLOGFILE"
      $THISCMD < "$INFILE"
      rc=$?
      if [ $rc != 0 ] ; then
         echo "ERROR:  The following command failed with return code $rc: "
         echo "        $THISCMD < $INFILE"
         exit $rc
      fi
   fi
}

function checkversion(){
  # inputs: $_linkfrom, $_cname, $CMDLOGFILE, $_version
  # output: $SKIPBUILD
  if [ -L "$_linkfrom" ] ; then 
    existing_install_dir=$(readlink "$_linkfrom")
    if [ -d "$existing_install_dir" ] ; then 
      existing_version=${existing_install_dir##*-} 
      if [ "$existing_version" == "$_version" ] ; then 
        echo "Info: Skipping build for $_cname, version $_version already exists" 
        echo "# skipping build for $_cname, version $_version already exists" >>"$CMDLOGFILE"
        SKIPBUILD=TRUE
      else
        echo "Info: creating new version of $_cname $_version"
        echo "Info: creating new version of $_cname $_version" >>"$CMDLOGFILE"
      fi
    else
      echo "Info: Missing existing_install_dir $existing_install_dir, creating version of $_cname $_version"
      echo "# Missing existing_install_dir $existing_install_dir, creating version of $_cname $_version" >>"$CMDLOGFILE"
    fi
  fi
}

function buildninja(){
  _cname="ninja"
  _version=1.13.2
  _installdir=$SROCK_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$SROCK_SUPP/$_cname
  _builddir=$SROCK_SUPP_BUILD/$_cname

  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then
    return
  fi
  if [ -d "$_builddir" ] ; then
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "wget https://github.com/ninja-build/ninja/archive/refs/tags/v${_version}.tar.gz"
  runcmd "tar -xzf v${_version}.tar.gz"
  runcmd "cd ninja-$_version"
  _patch_file="$thisdir/patches/ninja-nprocs-v${_version}.patch"
  if [ -r "$_patch_file" ]; then
    runcmd   "cp $_patch_file $_builddir"
    runcmdin "patch --merge -p1" "$_patch_file"
  fi
  if [ -d "$_installdir" ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir/bin"
  runcmd "$SROCK_SUPP/cmake/bin/cmake -Bbuild-cmake"
  runcmd "$SROCK_SUPP/cmake/bin/cmake --build build-cmake"
  runcmd "cp -p build-cmake/ninja $_installdir/bin/."
  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

function getrocmpackage(){
  if [[ "$1" == "" || "$2" ==  "" || "$3" == "" ]]; then
    echo "ERROR: getrocmpackage requires 3 parameters - localname packagename componentversion"
    exit 1
  fi
  _cname="$1"
  _packagename="$2"
  _componentversion="$3"
  _directory=$(echo "$2" | cut -b 1)
  _version=7.0
  _packageversion=7.0.0
  _fullversion=70000
  _buildnumber=38
  _installdir=$SROCK_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$SROCK_SUPP/$_cname
  _builddir=$SROCK_SUPP_BUILD/$_cname

  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then
    return
  fi
  if [ -d "$_builddir" ] ; then
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  osname=$(grep -e ^NAME= < /etc/os-release)
  if [[ $osname =~ "Ubuntu" ]]; then
    # not sure if deb_version is 20 or 22
    deb_version="24"
    os_version=$(grep VERSION_ID /etc/os-release | cut -d"\"" -f2)
    [ "$os_version" == "22.04" ] && deb_version="22"
    #https://repo.radeon.com/rocm/apt/6.1/pool/main/h/hsa-amd-aqlprofile6.1.0/hsa-amd-aqlprofile6.1.0_1.0.0.60100.60100-82~${deb_version}_amd64.deb
    #https://repo.radeon.com/rocm/apt/6.1/pool/main/h/hsa-amd-aqlprofile6.1.0/hsa-amd-aqlprofile6.1.0_1.0.0.60100.60100-82~22.04_amd64.deb
    runcmd "wget https://repo.radeon.com/rocm/apt/$_version/pool/main/$_directory/$_packagename$_packageversion/$_packagename${_packageversion}_${_componentversion}.${_fullversion}-${_buildnumber}~${deb_version}.04_amd64.deb"

    runcmd "dpkg -x $_packagename${_packageversion}_${_componentversion}.${_fullversion}-${_buildnumber}~${deb_version}.04_amd64.deb $_builddir"
  elif [[ $osname =~ "SLES" ]]; then
    #https://repo.radeon.com/rocm/yum/6.1/main/hsa-amd-aqlprofile6.1.0-1.0.0.60100.60100-82.el7.x86_64.rpm
    runcmd "wget https://repo.radeon.com/rocm/zyp/$_version/main/$_packagename$_packageversion-$_componentversion.$_fullversion-sles156.$_buildnumber.x86_64.rpm"
    echo "$_packagename$_packageversion-$_componentversion.$_fullversion-sles156.$_buildnumber.x86_64.rpm | cpio -idm"
    rpm2cpio "$_packagename$_packageversion-$_componentversion.$_fullversion-sles156.$_buildnumber.x86_64.rpm" | cpio -idm
  else
    runcmd "wget https://repo.radeon.com/rocm/rhel8/$_version/main/$_packagename$_packageversion-$_componentversion.$_fullversion-$_buildnumber.el8.x86_64.rpm"
    echo "$_packagename$_packageversion-$_componentversion.$_fullversion-$_buildnumber.el8.x86_64.rpm | cpio -idm"
    rpm2cpio "$_packagename$_packageversion-$_componentversion.$_fullversion-$_buildnumber.el8.x86_64.rpm" | cpio -idm
  fi

  if [ -d "$_installdir" ] ; then
    runcmd "rm -rf $_installdir"
  fi
  if [ "$_cname" == "rocm-core" ] ; then
    runcmd "mkdir -p $_installdir"
    runcmd "cp -rp $_builddir/opt/rocm-$_packageversion/. $_installdir"
  else
    runcmd "mkdir -p $_installdir/lib"
    runcmd "cd $_installdir"
    runcmd "cp -rp $_builddir/opt/rocm-$_packageversion/lib  $_installdir"
  fi

  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  #runcmd "rm -rf $_builddir"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

function buildcmake(){
  _cname="cmake"
  _version=3.31.0
  _installdir=$SROCK_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$SROCK_SUPP/$_cname
  _builddir=$SROCK_SUPP_BUILD/$_cname 
  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then 
    return
  fi

  if [ -d "$_builddir" ] ; then
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "wget https://github.com/Kitware/CMake/releases/download/v$_version/cmake-$_version.tar.gz"
  runcmd "tar -xzf cmake-$_version.tar.gz"
  runcmd "cd cmake-$_version"
  if [ -d "$_installdir" ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "./bootstrap --parallel=8 --prefix=$_installdir"
  runcmd "make -j8"
  runcmd "make install"
  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

#---------------------------   Main script starts here -----------------------

CMDLOGFILE=$SROCK_SUPP_BUILD/cmdlog

mkdir -p "$SROCK_SUPP_BUILD"
buildcmake
buildninja

echo "SROCK_SUPP=$SROCK_SUPP"
echo "SROCK_JOB_THREADS=$SROCK_JOB_THREADS"
echo "NINJA_NPROCS=$NINJA_NPROCS"
_thisdate=$(date)
echo "# DONE: successful build of cmake on $_thisdate " >>"$CMDLOGFILE"

