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
#
EOF
}

SUPPLEMENTAL_COMPONENTS=${SUPPLEMENTAL_COMPONENTS:-openmpi silo hdf5 fftw ninja}
PREREQUISITE_COMPONENTS=${PREREQUISITE_COMPONENTS:-cmake rocmsmilib hwloc aqlprofile}

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

function runcmd(){
   THISCMD=$1
   if [ $DRYRUN ] ; then
      echo "$THISCMD"
   else
      echo "$THISCMD" 
      echo "$THISCMD" >>$CMDLOGFILE
      $THISCMD
      rc=$?
      if [ $rc != 0 ] ; then
         echo "ERROR:  The following command failed with return code $rc: "
         echo "        $THISCMD"
         exit $rc
      fi
   fi
}

function checkversion(){
  # inputs: $_linkfrom, $_cname, $CMDLOGFILE, $_version
  # output: $SKIPBUILD
  if [ -L $_linkfrom ] ; then 
    existing_install_dir=`ls -l $_linkfrom | grep $_cname | awk '{print $11}'`
    if [ -d $existing_install_dir ] ; then 
      existing_version=${existing_install_dir##*-} 
      if [ "$existing_version" == "$_version" ] ; then 
        echo "Info: Skipping build for $_cname, version $_version already exists" 
        echo "# skipping build for $_cname, version $_version already exists" >>$CMDLOGFILE
        SKIPBUILD=TRUE
      else
        echo "Info: creating new version of $_cname $_version"
        echo "Info: creating new version of $_cname $_version" >>$CMDLOGFILE
      fi
    else
      echo "Info: Missing existing_install_dir $existing_install_dir, creating version of $_cname $_version"
      echo "# Missing existing_install_dir $existing_install_dir, creating version of $_cname $_version" >>$CMDLOGFILE
    fi
  fi
}
function buildopenmpi(){
  _cname="openmpi"
  _version=5.0.0
  _release=v5.0
  _installdir=$AOMP_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$AOMP_SUPP/$_cname
  _builddir=$AOMP_SUPP_BUILD/$_cname

  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then 
    return
  fi
  if [ -d $_builddir ] ; then 
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "wget https://download.open-mpi.org/release/open-mpi/$_release/openmpi-$_version.tar.bz2"
  runcmd "bzip2 -d openmpi-$_version.tar.bz2"
  runcmd "tar -xf openmpi-$_version.tar"
  runcmd "cd openmpi-$_version"
  if [ -d $_installdir ] ; then 
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "./configure --with-hwloc=$HOME/local/hwloc --with-hwloc-libdir=$HOME/local/hwloc/lib OMPI_CC=$AOMP/bin/clang OMPI_CXX=$AOMP/bin/clang++ OMPI_F90=$AOMP/bin/flang CXX=$AOMP/bin/clang++ CC=$AOMP/bin/clang FC=$AOMP/bin/flang --prefix=$_installdir"
  runcmd "make -j8"
  runcmd "make install"
  if [ -L $_linkfrom ] ; then 
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>$CMDLOGFILE
}

function buildninja(){
  _cname="ninja"
  _version=1.11.1
  _installdir=$AOMP_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$AOMP_SUPP/$_cname
  _builddir=$AOMP_SUPP_BUILD/$_cname

  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then
    return
  fi
  if [ -d $_builddir ] ; then
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "wget https://github.com/ninja-build/ninja/archive/refs/tags/v${_version}.tar.gz"
  runcmd "tar -xzf v${_version}.tar.gz"
  runcmd "cd ninja-$_version"
  if [ -d $_installdir ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir/bin"
  runcmd "$AOMP_SUPP/cmake/bin/cmake -Bbuild-cmake"
  runcmd "$AOMP_SUPP/cmake/bin/cmake --build build-cmake"
  runcmd "cp -p build-cmake/ninja $_installdir/bin/."
  if [ -L $_linkfrom ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>$CMDLOGFILE
}

function buildaqlprofile(){
  _cname="aqlprofile"
  _version=6.1
  _packageversion=6.1.0
  _fullversion=60100
  _buildnumber=82
  _installdir=$AOMP_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$AOMP_SUPP/$_cname
  _builddir=$AOMP_SUPP_BUILD/$_cname

  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then
    return
  fi
  if [ -d $_builddir ] ; then
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  osname=$(cat /etc/os-release | grep -e ^NAME=)
  if [[ $osname =~ "Ubuntu" ]]; then
    # not sure if deb_version is 20 or 22
    deb_version="22"
    os_version=`grep VERSION_ID /etc/os-release | cut -d"\"" -f2`
    [ $os_version == "20.04" ] && deb_version="20"
    #https://repo.radeon.com/rocm/apt/6.1/pool/main/h/hsa-amd-aqlprofile6.1.0/hsa-amd-aqlprofile6.1.0_1.0.0.60100.60100-82~${deb_version}_amd64.deb
    #https://repo.radeon.com/rocm/apt/6.1/pool/main/h/hsa-amd-aqlprofile6.1.0/hsa-amd-aqlprofile6.1.0_1.0.0.60100.60100-82~22.04_amd64.deb
    runcmd "wget https://repo.radeon.com/rocm/apt/"$_version"/pool/main/h/hsa-amd-aqlprofile"$_packageversion"/hsa-amd-aqlprofile"$_packageversion"_1.0.0."$_fullversion"."$_fullversion"-"$_buildnumber"~${deb_version}.04_amd64.deb"

    runcmd "dpkg -x hsa-amd-aqlprofile"$_packageversion"_1.0.0."$_fullversion"."$_fullversion"-"$_buildnumber"~${deb_version}.04_amd64.deb $_builddir"
  elif [[ $osname =~ "SLES" ]]; then
    #https://repo.radeon.com/rocm/yum/6.1/main/hsa-amd-aqlprofile6.1.0-1.0.0.60100.60100-82.el7.x86_64.rpm
    runcmd "wget https://repo.radeon.com/rocm/zyp/"$_version"/main/hsa-amd-aqlprofile"$_packageversion"-1.0.0."$_fullversion"."$_fullversion"-sles154."$_buildnumber".x86_64.rpm"
    echo "hsa-amd-aqlprofile"$_packageversion"-1.0.0."$_fullversion"."$_fullversion"-sles154."$_buildnumber".x86_64.rpm | cpio -idm"
    rpm2cpio hsa-amd-aqlprofile"$_packageversion"-1.0.0."$_fullversion"."$_fullversion"-sles154."$_buildnumber".x86_64.rpm | cpio -idm
  else
    runcmd "wget https://repo.radeon.com/rocm/yum/"$_version"/main/hsa-amd-aqlprofile"$_packageversion"-1.0.0."$_fullversion"."$_fullversion"-"$_buildnumber".el7.x86_64.rpm"
    echo "hsa-amd-aqlprofile"$_packageversion"-1.0.0."$_fullversion"."$_fullversion"-"$_buildnumber".el7.x86_64.rpm | cpio -idm"
    rpm2cpio hsa-amd-aqlprofile"$_packageversion"-1.0.0."$_fullversion"."$_fullversion"-"$_buildnumber".el7.x86_64.rpm | cpio -idm
  fi

  if [ -d $_installdir ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir/lib"
  runcmd "cd $_installdir"
  runcmd "cp -rp $_builddir/opt/rocm-"$_packageversion"/lib  $_installdir"

  if [ -L $_linkfrom ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  #runcmd "rm -rf $_builddir"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>$CMDLOGFILE
}

function buildhdf5(){
  _cname="hdf5"
  _version=1.12.0
  _release=hdf5-1.12
  _installdir=$AOMP_SUPP_INSTALL/hdf5-$_version
  _linkfrom=$AOMP_SUPP/hdf5
  _builddir=$AOMP_SUPP_BUILD/hdf5
  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then 
    return
  fi

  if [ -d $_builddir ] ; then 
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd " wget https://support.hdfgroup.org/ftp/HDF5/releases/$_release/hdf5-$_version/src/hdf5-$_version.tar.bz2"
  runcmd "bzip2 -d hdf5-$_version.tar.bz2"
  runcmd "tar -xf hdf5-$_version.tar"
  runcmd "cd hdf5-$_version"
  if [ -d $_installdir ] ; then 
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "./configure --enable-fortran --prefix=$_installdir"
  runcmd "make -j8"
  runcmd "make install"
  if [ -L $_linkfrom ] ; then 
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>$CMDLOGFILE
}

function buildsilo(){
  _cname="silo"
  _version=4.10.2
  _installdir=$AOMP_SUPP_INSTALL/silo-$_version
  _linkfrom=$AOMP_SUPP/silo
  _builddir=$AOMP_SUPP_BUILD/silo
  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then 
    return
  fi

  if [ -d $_builddir ] ; then 
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  # runcmd "wget https://wci.llnl.gov/sites/wci/files/2021-01/silo-$_version.tgz"
  # runcmd "tar -xzf silo-$_version.tgz"
  runcmd "wget https://software.llnl.gov/Silo/ghpages/releases/silo-$_version.tar.xz"
  runcmd "tar -x --xz -f silo-$_version.tar.xz"
  runcmd "cd silo-$_version"
  if [ -d $_installdir ] ; then 
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "./configure --prefix=$_installdir"
  runcmd "make -j8"
  runcmd "make install"
  if [ -L $_linkfrom ] ; then 
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>$CMDLOGFILE
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

  if [ -d $_builddir ] ; then
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "wget http://www.fftw.org/fftw-$_version.tar.gz"
  runcmd "tar -xzf fftw-$_version.tar.gz"
  runcmd "cd fftw-$_version"
  if [ -d $_installdir ] ; then
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "./configure --prefix=$_installdir --enable-shared --enable-threads --enable-sse2 --enable-avx"
  runcmd "make -j8"
  runcmd "make install"
  runcmd "make clean"
  runcmd "./configure --prefix=$_installdir --enable-shared --enable-threads --enable-sse2 --enable-avx --enable-float"
  runcmd "make -j8"
  runcmd "make install"
  if [ -L $_linkfrom ] ; then
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>$CMDLOGFILE
}


function buildcmake(){
  _cname="cmake"
  _version=3.22.1
  _installdir=$AOMP_SUPP_INSTALL/$_cname-$_version
  _linkfrom=$AOMP_SUPP/$_cname
  _builddir=$AOMP_SUPP_BUILD/$_cname 
  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then 
    return
  fi

  if [ -d $_builddir ] ; then 
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "wget https://github.com/Kitware/CMake/releases/download/v$_version/cmake-$_version.tar.gz"
  runcmd "tar -xzf cmake-$_version.tar.gz"
  runcmd "cd cmake-$_version"
  if [ -d $_installdir ] ; then 
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "./bootstrap --prefix=$_installdir"
  runcmd "make -j8"
  runcmd "make install"
  if [ -L $_linkfrom ] ; then 
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>$CMDLOGFILE
}

function buildrocmsmilib(){
  _cname="rocmsmilib"
  _version=6.1.x
  _installdir=$AOMP_SUPP_INSTALL/rocmsmilib-$_version
  _linkfrom=$AOMP_SUPP/rocmsmilib
  _builddir=$AOMP_SUPP_BUILD/rocmsmilib
  SKIPBUILD="FALSE"
  checkversion
  if [ "$SKIPBUILD" == "TRUE"  ] ; then 
    return
  fi

  if [ -d $_builddir ] ; then 
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "git clone -b rocm-$_version https://github.com/ROCm/rocm_smi_lib rocmsmilib-$_version"
  runcmd "cd rocmsmilib-$_version"
  runcmd "mkdir -p build"
  runcmd "cd build"
  runcmd "$AOMP_SUPP/cmake/bin/cmake -DCMAKE_INSTALL_PREFIX=$_installdir .."
  if [ -d $_installdir ] ; then 
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "make -j8"
  runcmd "make install"
  if [ -L $_linkfrom ] ; then 
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>$CMDLOGFILE
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

  if [ ! -d $AOMP_SUPP/rocmsmilib/lib ] && [ ! -d $AOMP_SUPP/rocmsmilib/lib64 ]; then
    echo "ERROR: Must build rocmsmilib before hwloc. Try:"
    echo "       $0 rocmsmilib"
    echo "#ERROR: You must build rocmsmilib before hwloc because static build of hwloc depends on rocsmilib">>$CMDLOGFILE
    exit 1
  fi
  if [ -d $_builddir ] ; then 
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "git clone https://github.com/open-mpi/hwloc hwloc-$_version"
  runcmd "cd hwloc-$_version"
  runcmd "git checkout v$_version"
  runcmd "./autogen.sh"
  runcmd "./configure --prefix=$_installdir --with-pic=yes --enable-static=yes --enable-shared=no --disable-io --disable-libudev --disable-libxml2 --with-rocm=$AOMP_SUPP/rocsmilib"
  if [ -d $_installdir ] ; then 
    runcmd "rm -rf $_installdir"
  fi
  runcmd "mkdir -p $_installdir"
  runcmd "make -j8"
  runcmd "make install"
  if [ -L $_linkfrom ] ; then 
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>$CMDLOGFILE
}

#---------------------------   Main script starts here -----------------------
sname=${0##*/} 
CMDLOGFILE=$AOMP_SUPP_BUILD/cmdlog
mkdir -p $AOMP_SUPP_BUILD
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
  _components=$@
fi
# save the current directory
curdir=$PWD
for _component in $_components ; do 
  _thisdate=`date`
  echo "" >>$CMDLOGFILE
  echo "# -------------------------------------------------" >>$CMDLOGFILE
  echo "# $_component build started on $_thisdate" >>$CMDLOGFILE
  if [ $_component == "openmpi" ] ; then
    buildopenmpi
  elif [ $_component == "silo" ] ; then
    buildsilo
  elif [ $_component == "hdf5" ] ; then
    buildhdf5
  elif [ $_component == "fftw" ] ; then
    buildfftw
  elif [ $_component == "hwloc" ] ; then
    buildhwloc
  elif [ $_component == "cmake" ] ; then
    buildcmake
  elif [ $_component == "rocmsmilib" ] ; then
    buildrocmsmilib
  elif [ $_component == "ninja" ] ; then
    buildninja
  elif [ $_component == "aqlprofile" ] ; then
    buildaqlprofile
  else
    echo "ERROR:  Invalid component name $_component" >>$CMDLOGFILE
    echo "ERROR:  Invalid component name $_component"
    if [ "$sname" == "build_prereq.sh" ] ; then
       echo "        Must be a subset of: $PREREQUISITE_COMPONENTS"
    else
       echo "        Must be a subset of: $SUPPLEMENTAL_COMPONENTS"
    fi
    exit 0
  fi
  _thisdate=`date`
  echo "# DONE: successful build of $_component on $_thisdate " >>$CMDLOGFILE
done
# restore the current directory
cd $curdir
