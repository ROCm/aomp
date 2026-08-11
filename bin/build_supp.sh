#!/bin/bash
function build_supp_help(){
/bin/cat 2>&1 <<"EOF"
#
#  Component system for AOMP developers and testers for both
#  prerequisit and supplemental components.
#  These are different that compiler build components such as libdevice
#  in that artifacts from these components are not installed into the
#  AOMP installation.  These components are not typically need by end
#  users but are needed to build the compiler or to test the compiler.
# 
#  build_supp.sh : Script to build Supplemental components for compiler
#                  testing such as openmpi, hdf5, and silo.
#                  If used with no arg, all supplemental components
#                  listed by SUPPLEMENTAL_COMPONENTS will be built.
#                  If a single arg is given only that component will
#                  be built.
#
#  build_prereq.sh: Script to build prerequisite components for building
#                  the aomp compiler. This is symbolic link to build_supp.sh
#                  If used with no arg, all prerequisiste components
#                  listed by PREREQUISITE_COMPONENTS will be built.
#                  If a single arg is given only that component will
#                  be built.
#
# Applications or AOMP build scripts that need supplemental components 
# can locate the latest version with $AOMP_SUPP/<component name>
# The default value for AOMP_SUPP is $HOME/local.
# Supplemental components are built with either the ROCm or AOMP
# compiler using this script. This script uses the AOMP environment
# variable to identify which LLVM to use.
#
# Directory structure for supplemental and prerequisite components:
#
# $AOMP_SUPP                           Base directory for all components
# $AOMP_SUPP/install/<cname>-<version> Download directory for version <version>
#                                      of component <cname>.
# $AOMP_SUPP/build/<cname>             Build directory for component <cname>
$ $AOMP_SUPP/build/cmdlog              File with log of all components built
# $AOMP_SUPP/<cname>                   Symbolic link to last intall directory
#                                      of component <cname>
#
# AOMP scripts that use component <cname> should find the installation
# in directory $AOMP_SUPP/<cname>.  For example the openmpi lib directory
# should be referenced as $AOMP_SUPP/<cname>/lib
# 
# Known issues:
# - The _version name for each component must NOT have a "-" in it
#   because that is used to parse the version from the symbolic link.
# - One cannot build openmpi till one builds flang and installs into $AOMP. 
#
EOF
}

SUPPLEMENTAL_COMPONENTS=${SUPPLEMENTAL_COMPONENTS:-openmpi silo hdf5 fftw ninja rocmopenmpi xpmem ucx ucc}
PREREQUISITE_COMPONENTS=${PREREQUISITE_COMPONENTS:-cmake rocmsmilib hwloc aqlprofile rocm-core}

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/aomp_common_vars"
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
    existing_install_dir=$(readlink -f "$_linkfrom")
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

function derive_rocm_path(){
  # Derive ROCM_PATH - for AOMP installations, AOMP itself is the ROCm root
  # Check if AOMP has ROCm headers (include/hip, include/rocm-core, etc.)
  if [ -d "$AOMP/include/hip" ] || [ -d "$AOMP/include/rocm-core" ] ; then
    ROCM_PATH=$AOMP
  elif [ -d "$AOMPHIP/include/hip" ] || [ -d "$AOMPHIP/include/rocm-core" ] ; then
    ROCM_PATH=$AOMPHIP
  elif [ -n "$LLVM_INSTALL_LOC" ] && [ -d "$LLVM_INSTALL_LOC/../../../include/hip" ] ; then
    # For standard ROCm installations: LLVM at $ROCM/lib/llvm
    ROCM_PATH=$(realpath "$LLVM_INSTALL_LOC/../../..")
  elif [ -d "$(realpath -m "$AOMP")/../../include/hip" ] ; then
    # Fallback: check parent of AOMP
    ROCM_PATH=$(realpath -m "$(realpath -m "$AOMP")"/../..)
  else
    echo "Error: Cannot determine ROCM_PATH."
    echo "       Expected ROCm headers at \$AOMP/include/hip or similar."
    echo "       AOMP=$AOMP"
    return 1
  fi
  ROCM_PATH=$(realpath "$ROCM_PATH")
  export HIPCC="$ROCM_PATH/bin/amdclang"
  return 0
}

################################################################################
# XPMEM - Cross-Process Memory Access for high-performance shared memory
################################################################################
function buildxpmem(){
  _cname="xpmem"
  _version=2.7.4
  _installdir=$AOMP_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$AOMP_SUPP/$_cname
  _builddir=$AOMP_SUPP_BUILD/$_cname

  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE" ] ; then
    return
  fi
  if [ -d "$_builddir" ] ; then
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "wget https://github.com/openucx/xpmem/archive/refs/tags/v$_version.tar.gz"
  runcmd "tar -xzf v$_version.tar.gz"
  runcmd "cd xpmem-$_version"
  if [ -d "$_installdir" ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "./autogen.sh"
  runcmd "./configure --prefix=$_installdir"
  runcmd "make -j${AOMP_JOB_THREADS}"
  runcmd "make install"
  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sfr $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

################################################################################
# UCX - Unified Communication X for high-performance networking
################################################################################
function builducx(){
  _cname="ucx"
  _version=1.20.0
  _installdir=$AOMP_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$AOMP_SUPP/$_cname
  _builddir=$AOMP_SUPP_BUILD/$_cname

  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then
    return
  fi

  derive_rocm_path || return

  # Check if XPMEM is available
  if [ ! -d "$AOMP_SUPP/xpmem" ] ; then
    echo "Info: XPMEM not found at $AOMP_SUPP/xpmem, building it first..."
    buildxpmem
  fi
  XPMEM_PATH=$AOMP_SUPP/xpmem

  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE" ] ; then
    return
  fi
  if [ -d "$_builddir" ] ; then
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "wget https://github.com/openucx/ucx/releases/download/v$_version/ucx-$_version.tar.gz"
  runcmd "tar -xzf ucx-$_version.tar.gz"
  runcmd "cd ucx-$_version"
  runcmd "mkdir -p build"
  runcmd "cd build"
  if [ -d "$_installdir" ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"

  # Configure UCX with ROCm and XPMEM support
  runcmd "../contrib/configure-release \
    --prefix=$_installdir \
    --with-rocm=$ROCM_PATH \
    --with-xpmem=$XPMEM_PATH \
    --without-cuda \
    --enable-mt \
    --enable-optimizations \
    --disable-logging \
    --disable-debug \
    --enable-assertions \
    --enable-params-check \
    --enable-examples"

  runcmd "make -j${AOMP_JOB_THREADS}"
  runcmd "make install"
  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sfr $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

################################################################################
# UCC - Unified Collective Communication for collective operations
################################################################################
function builducc(){
  _cname="ucc"
  _version=1.6.0
  _installdir=$AOMP_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$AOMP_SUPP/$_cname
  _builddir=$AOMP_SUPP_BUILD/$_cname

  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then
    return
  fi

  derive_rocm_path || return

  # Check if UCX is available
  if [ ! -d "$AOMP_SUPP/ucx" ] ; then
    echo "Info: UCX not found at $AOMP_SUPP/ucx, building it first..."
    builducx
  fi
  UCX_PATH=$AOMP_SUPP/ucx

  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE" ] ; then
    return
  fi
  if [ -d "$_builddir" ] ; then
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "wget https://github.com/openucx/ucc/archive/refs/tags/v$_version.tar.gz"
  runcmd "tar -xzf v$_version.tar.gz"
  runcmd "cd ucc-$_version"
  if [ -d "$_installdir" ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "./autogen.sh"

  # Configure UCC with ROCm and UCX support
  runcmd "./configure \
    --prefix=$_installdir \
    --with-rocm=$ROCM_PATH \
    --with-ucx=$UCX_PATH"

  runcmd "make -j${AOMP_JOB_THREADS}"
  runcmd "make install"
  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sfr $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

################################################################################
# OpenMPI build helper - shared infrastructure for openmpi and rocmopenmpi
# Usage: _buildopenmpi_impl <cname> <version> [extra_configure_opts...]
################################################################################
function _buildopenmpi_impl(){
  local _cname="$1"
  local _version="$2"
  shift 2
  local _extra_configure_opts="$*"
  local _release=v5.0
  local _installdir=$AOMP_SUPP_INSTALL/$_cname-$_version
  local _linkfrom=$AOMP_SUPP/$_cname
  local _builddir=$AOMP_SUPP_BUILD/$_cname

  # Not all builds, trunk for example, install clang into lib/llvm/bin. Fall back on $AOMP/bin.
  if [ ! -f "$LLVM_INSTALL_LOC/bin/${FLANG}" ] ; then
    LLVM_INSTALL_LOC=$AOMP
    if [ ! -f "$LLVM_INSTALL_LOC/bin/${FLANG}" ] ; then
      LLVM_INSTALL_LOC=$AOMP/lib/llvm
      if [ ! -f "$LLVM_INSTALL_LOC/bin/${FLANG}" ] ; then
        echo "Error: $_cname build cannot find ${FLANG} executable. Set AOMP to location of $FLANG "
        exit 1
      fi
    fi
  fi
  if [ ! -d "$AOMP_SUPP/hwloc" ] ; then
    echo "Error: 'build_supp.sh $_cname' requires that hwloc is installed at $AOMP_SUPP/hwloc"
    echo "       Please run 'build_supp.sh hwloc' "
    exit 1
  fi

  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE" ] ; then
    return
  fi

  if [ -d "$_builddir" ] ; then
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "wget https://download.open-mpi.org/release/open-mpi/$_release/openmpi-$_version.tar.bz2"
  runcmd "bzip2 -d openmpi-$_version.tar.bz2"
  runcmd "tar -xf openmpi-$_version.tar"
  runcmd "cd openmpi-$_version"
  if [ -d "$_installdir" ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"

  # Update configure to recognize flang
  runcmd "cp configure configure-orig"
  runcmdout "sed -e s/flang\s*)/flang*)/ configure-orig" configure

  # Configure with common options plus any extra options
  runcmd "./configure \
    --prefix=$_installdir \
    --with-hwloc=$AOMP_SUPP/hwloc \
    --with-hwloc-libdir=$AOMP_SUPP/hwloc/lib \
    OMPI_CC=$LLVM_INSTALL_LOC/bin/clang \
    OMPI_CXX=$LLVM_INSTALL_LOC/bin/clang++ \
    OMPI_F90=$LLVM_INSTALL_LOC/bin/${FLANG} \
    CXX=$LLVM_INSTALL_LOC/bin/clang++ \
    CC=$LLVM_INSTALL_LOC/bin/clang \
    FC=$LLVM_INSTALL_LOC/bin/${FLANG} \
    $_extra_configure_opts"

  if [[ "$_version" == "5.0.10" ]] ; then
    if [ -f "$thisdir/patches/ompi.patch" ] ; then
       runcmdin "patch --merge -p1" "$thisdir/patches/ompi.patch"
    fi
  fi

  runcmd "make -j${AOMP_JOB_THREADS}"
  runcmd "make install"
  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sfr $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

################################################################################
# OpenMPI (standard build without ROCm support)
################################################################################
function buildopenmpi(){
  _cname="openmpi"
  _version=5.0.8
  _buildopenmpi_impl $_cname $_version
}

################################################################################
# ROCm OpenMPI - OpenMPI with ROCm/GPU-aware MPI support
# This builds OpenMPI with UCX, UCC, and ROCm support for GPU-aware MPI
################################################################################
function buildrocmopenmpi(){
  _cname="rocmopenmpi"
  _version=5.0.10
  _installdir=$AOMP_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$AOMP_SUPP/$_cname
  _builddir=$AOMP_SUPP_BUILD/$_cname

  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then
    return
  fi

  derive_rocm_path || return
  echo "Info: Using ROCM_PATH=$ROCM_PATH"

  # Check and build dependencies if needed
  if [ ! -d "$AOMP_SUPP/ucx" ] ; then
    echo "Info: UCX not found at $AOMP_SUPP/ucx, building it first..."
    builducx
  fi
  UCX_PATH=$AOMP_SUPP/ucx

  if [ ! -d "$AOMP_SUPP/ucc" ] ; then
    echo "Info: UCC not found at $AOMP_SUPP/ucc, building it first..."
    builducc
  fi
  UCC_PATH=$AOMP_SUPP/ucc

  # Build OpenMPI with ROCm-specific configure options
  _buildopenmpi_impl $_cname $_version \
    "--with-rocm=$ROCM_PATH" \
    "--with-ucx=$UCX_PATH" \
    "--with-ucc=$UCC_PATH" \
    "--enable-mca-no-build=btl-uct" \
    "--enable-mpi" \
    "--enable-mpi-fortran" \
    "--disable-debug"

  # Configure default MCA parameters for UCX
  local _installdir=$AOMP_SUPP_INSTALL/rocmopenmpi-5.0.10
  if [ -d "$_installdir/etc" ] ; then
    echo "# Setting UCX as default point-to-point and one-sided communication"
    {
      echo "pml = ucx"
      echo "osc = ucx"
      echo "coll_ucc_enable = 1"
      echo "coll_ucc_priority = 100"
    } >> "${_installdir}/etc/openmpi-mca-params.conf"
    echo "# MCA params configured for UCX default" >>"$CMDLOGFILE"
  fi
}
function buildninja(){
  _cname="ninja"
  _version=1.13.2
  _installdir=$AOMP_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$AOMP_SUPP/$_cname
  _builddir=$AOMP_SUPP_BUILD/$_cname

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
  runcmd "$AOMP_SUPP/cmake/bin/cmake -Bbuild-cmake"
  runcmd "$AOMP_SUPP/cmake/bin/cmake --build build-cmake"
  runcmd "cp -p build-cmake/ninja $_installdir/bin/."
  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sfr $_installdir $_linkfrom"
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
  _version=7.1
  _packageversion=7.1.0
  _fullversion=70100
  _buildnumber=20
  _installdir=$AOMP_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$AOMP_SUPP/$_cname
  _builddir=$AOMP_SUPP_BUILD/$_cname

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
    runcmd "wget https://repo.radeon.com/rocm/apt/$_version/pool/main/$_directory/$_packagename$_packageversion/$_packagename${_packageversion}_${_componentversion}.${_fullversion}-${_buildnumber}~${deb_version}.04_amd64.deb"
    runcmd "dpkg -x $_packagename${_packageversion}_${_componentversion}.${_fullversion}-${_buildnumber}~${deb_version}.04_amd64.deb $_builddir"
  elif [[ $osname =~ "SLES" ]]; then
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
  runcmd "ln -sfr $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

function buildhdf5(){
  _cname="hdf5"
  _version=1.14.0
  _release=hdf5-1.14
  _installdir=$AOMP_SUPP_INSTALL/hdf5-$_version
  _linkfrom=$AOMP_SUPP/hdf5
  _builddir=$AOMP_SUPP_BUILD/hdf5
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
  runcmd " wget https://support.hdfgroup.org/ftp/HDF5/releases/$_release/hdf5-$_version/src/hdf5-$_version.tar.bz2"
  runcmd "bzip2 -d hdf5-$_version.tar.bz2"
  runcmd "tar -xf hdf5-$_version.tar"
  runcmd "cd hdf5-$_version"
  if [ -d "$_installdir" ] ; then 
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "./configure --enable-fortran --prefix=$_installdir"
  runcmd "make -j${AOMP_JOB_THREADS}"
  runcmd "make install"
  if [ -L "$_linkfrom" ] ; then 
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sfr $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

function buildsilo(){
  _cname="silo"
  _version=4.11.1
  _installdir=$AOMP_SUPP_INSTALL/silo-$_version
  _linkfrom=$AOMP_SUPP/silo
  _builddir=$AOMP_SUPP_BUILD/silo
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
  runcmd "wget https://github.com/LLNL/Silo/releases/download/$_version/silo-$_version.tar.xz"
  runcmd "tar -x --xz -f silo-$_version.tar.xz"
  runcmd "cd silo-$_version"
  if [ -d "$_installdir" ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "./configure --prefix=$_installdir"
  runcmd "make -j${AOMP_JOB_THREADS}"
  runcmd "make install"
  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sfr $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

function buildfftw(){
  _cname="fftw"
  _version=3.3.8
  _installdir=$AOMP_SUPP_INSTALL/fftw-$_version
  _linkfrom=$AOMP_SUPP/fftw
  _builddir=$AOMP_SUPP_BUILD/fftw
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
  runcmd "wget http://www.fftw.org/fftw-$_version.tar.gz"
  runcmd "tar -xzf fftw-$_version.tar.gz"
  runcmd "cd fftw-$_version"
  if [ -d "$_installdir" ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "./configure --prefix=$_installdir --enable-shared --enable-threads --enable-sse2 --enable-avx"
  runcmd "make -j${AOMP_JOB_THREADS}"
  runcmd "make install"
  runcmd "make clean"
  runcmd "./configure --prefix=$_installdir --enable-shared --enable-threads --enable-sse2 --enable-avx --enable-float"
  runcmd "make -j${AOMP_JOB_THREADS}"
  runcmd "make install"
  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sfr $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}


function buildcmake(){
  _cname="cmake"
  _version=3.31.11
  _installdir=$AOMP_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$AOMP_SUPP/$_cname
  _builddir=$AOMP_SUPP_BUILD/$_cname 
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
  runcmd "make -j${AOMP_JOB_THREADS}"
  runcmd "make install"
  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sfr $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

function buildrocmsmilib(){
  _cname="rocmsmilib"
  _version=7.1.x
  _installdir=$AOMP_SUPP_INSTALL/rocmsmilib-$_version
  _linkfrom=$AOMP_SUPP/rocmsmilib
  _builddir=$AOMP_SUPP_BUILD/rocmsmilib
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
  runcmd "git clone -b release/rocm-rel-7.1 https://github.com/ROCm/rocm_smi_lib rocmsmilib-$_version"
  runcmd "cd rocmsmilib-$_version"
  runcmd "mkdir -p build"
  runcmd "cd build"
  runcmd "$AOMP_SUPP/cmake/bin/cmake -DCMAKE_INSTALL_PREFIX=$_installdir .."
  if [ -d "$_installdir" ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "make -j${AOMP_JOB_THREADS}"
  runcmd "make install"
  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sfr $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

function buildhwloc(){
  _cname="hwloc"
  _version=2.7
  _installdir=$AOMP_SUPP_INSTALL/hwloc-$_version
  _linkfrom=$AOMP_SUPP/hwloc
  _builddir=$AOMP_SUPP_BUILD/hwloc
  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then 
    return
  fi

  if [ ! -d "$AOMP_SUPP/rocmsmilib/lib" ] && [ ! -d "$AOMP_SUPP/rocmsmilib/lib64" ]; then
    echo "ERROR: Must build rocmsmilib before hwloc. Try:"
    echo "       $0 rocmsmilib"
    echo "#ERROR: You must build rocmsmilib before hwloc because static build of hwloc depends on rocsmilib">>"$CMDLOGFILE"
    exit 1
  fi
  if [ -d "$_builddir" ] ; then
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "git clone https://github.com/open-mpi/hwloc hwloc-$_version"
  runcmd "cd hwloc-$_version"
  runcmd "git checkout v$_version"
  runcmd "./autogen.sh"
  runcmd "./configure --prefix=$_installdir --with-pic=yes --enable-static=yes --enable-shared=no --disable-io --disable-libudev --disable-libxml2 --with-rocm=$AOMP_SUPP/rocsmilib"
  if [ -d "$_installdir" ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "make -j${AOMP_JOB_THREADS}"
  runcmd "make install"
  if [ -L "$_linkfrom" ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sfr $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>"$CMDLOGFILE"
}

#---------------------------   Main script starts here -----------------------
sname=${0##*/} 
CMDLOGFILE=$AOMP_SUPP_BUILD/cmdlog
mkdir -p "$AOMP_SUPP_BUILD"
if [ "$1" == "-h" ] ; then 
  build_supp_help
  exit 0
fi
if [ "$1" == "install" ] ; then
  # build_aomp.sh will try to install each aomp component including "prereq"
  # but they are already installed so we just return here to avoid error message.
  # when building aomp from scratch.
  exit 0
fi
if [ "$1" == "" ] ; then 
  if [ "$sname" == "build_prereq.sh" ] ; then
    _components="$PREREQUISITE_COMPONENTS"
  else 
    _components="$SUPPLEMENTAL_COMPONENTS"
  fi
else
  _components=$*
fi
# save the current directory
curdir=$PWD
for _component in $_components ; do 
  _thisdate=$(date)
  {
    echo ""
    echo "# -------------------------------------------------"
    echo "# $_component build started on $_thisdate"
  } >> "$CMDLOGFILE"
  if [ "$_component" == "openmpi" ] ; then
    buildopenmpi
  elif [ "$_component" == "rocmopenmpi" ] ; then
    buildrocmopenmpi
  elif [ "$_component" == "xpmem" ] ; then
    buildxpmem
  elif [ "$_component" == "ucx" ] ; then
    builducx
  elif [ "$_component" == "ucc" ] ; then
    builducc
  elif [ "$_component" == "silo" ] ; then
    buildsilo
  elif [ "$_component" == "hdf5" ] ; then
    buildhdf5
  elif [ "$_component" == "fftw" ] ; then
    buildfftw
  elif [ "$_component" == "hwloc" ] ; then
    buildhwloc
  elif [ "$_component" == "cmake" ] ; then
    buildcmake
  elif [ "$_component" == "rocmsmilib" ] ; then
    buildrocmsmilib
  elif [ "$_component" == "ninja" ] ; then
    buildninja
  elif [ "$_component" == "aqlprofile" ] ; then
    getrocmpackage aqlprofile hsa-amd-aqlprofile 1.0.0
  elif [ "$_component" == "openclicdloader" ] ; then
    getrocmpackage openclicdloader rocm-opencl-icd-loader 1.2
  elif [ "$_component" == "rocm-core" ] ; then
    getrocmpackage rocm-core rocm-core 7.1.0
  else
    echo "ERROR:  Invalid component name $_component" >>"$CMDLOGFILE"
    echo "ERROR:  Invalid component name $_component"
    if [ "$sname" == "build_prereq.sh" ] ; then
       echo "        Must be a subset of: $PREREQUISITE_COMPONENTS"
    else
       echo "        Must be a subset of: $SUPPLEMENTAL_COMPONENTS"
    fi
    exit 0
  fi
  _thisdate=$(date)
  echo "# DONE: successful build of $_component on $_thisdate " >>"$CMDLOGFILE"
done

cd "$curdir" || exit
