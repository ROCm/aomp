#!/bin/bash
# 
#  build_supp.sh : Script to build Supplemental components for testing.
#                  such as openmpi, hdf5, and silo.
#                  If used with no arg, all supplemental components will
#                  be built. 
#
# Applications that need supplemental components can get the latest version 
# from $AOMP_SUPP/<component name>
# For example set OPENMPI_DIR=$AOMP_SUPP/openmpi in the run script.
# Supplemental components are built with either the ROCm or AOMP
# compiler using the build_supp.sh script. The build_supp.sh script
# will use the AOMP environment variable to identify which LLVM to use
# for building the supplemental component.
# The directory structure for a supplemental component includes:
#   1. a build directory for each component,
#   2. an install directory for each version of a component,
#   3. a link from AOMP_SUPP/<component name> to latest version.
# For example: The openmpi component is built in directory
# $AOMP_SUPP_BUILD/openmpi/ and installed in directory
# $AOMP_SUPP_INSTALL/openmpi/openmpi-4.1.1/ with a symbolic link from
# $AOMP_SUPP/openmpi to $AOMP_SUPP_INSTALL/openmpi/openmpi-4.1.1/
#

SUPPLEMENTAL_COMPONENTS="openmpi silo hdf5"

# --- Start standard header to set AOMP environment variables ----
function getdname(){
   local __DIRN=`dirname "$1"`
   if [ "$__DIRN" = "." ] ; then
      __DIRN=$PWD;
   else
      if [ ${__DIRN:0:1} != "/" ] ; then
         if [ ${__DIRN:0:2} == ".." ] ; then
               __DIRN=`dirname $PWD`/${__DIRN:3}
         else
            if [ ${__DIRN:0:1} = "." ] ; then
               __DIRN=$PWD/${__DIRN:2}
            else
               __DIRN=$PWD/$__DIRN
            fi
         fi
      fi
   fi
   echo $__DIRN
}
thisdir=$(getdname $0)
[ ! -L "$0" ] || thisdir=$(getdname `readlink "$0"`)
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

function buildopenmpi(){
  _version=4.1.1
  _release=v4.1
  _installdir=$AOMP_SUPP_INSTALL/openmpi-$_version
  _linkfrom=$AOMP_SUPP/openmpi
  _builddir=$AOMP_SUPP_BUILD/openmpi

  if [ -d $_builddir ] ; then 
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "wget https://download.open-mpi.org/release/open-mpi/$_release/openmpi-$_version.tar.bz2"
  runcmd "bzip2 -d openmpi-$_version.tar.bz2"
  runcmd "tar -xf openmpi-$_version.tar"
  runcmd "cd openmpi-$_version"
  runcmd "mkdir -p $_installdir"
  runcmd "./configure OMPI_CC=$AOMP/bin/clang OMPI_CXX=$AOMP/bin/clang++ OMPI_F90=$AOMP/bin/flang CXX=$AOMP/bin/clang++ CC=$AOMP/bin/clang FC=$AOMP/bin/flang --prefix=$_installdir"
  runcmd "make -j8"
  if [ -d $_installdir ] ; then 
    runcmd "rm -rf $_installdir"
  fi
  runcmd "make install"
  if [ -L $_linkfrom ] ; then 
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>$CMDLOGFILE
}

function buildhdf5(){
  _version=1.12.0
  _release=hdf5-1.12
  _installdir=$AOMP_SUPP_INSTALL/hdf5-$_version
  _linkfrom=$AOMP_SUPP/hdf5
  _builddir=$AOMP_SUPP_BUILD/hdf5

  if [ -d $_builddir ] ; then 
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd " wget https://support.hdfgroup.org/ftp/HDF5/releases/$_release/hdf5-$_version/src/hdf5-$_version.tar.bz2"
  runcmd "bzip2 -d hdf5-$_version.tar.bz2"
  runcmd "tar -xf hdf5-$_version.tar"
  runcmd "cd hdf5-$_version"
  runcmd "mkdir -p $_installdir"
  runcmd "./configure --enable-fortran --prefix=$_installdir"
  runcmd "make -j8"
  if [ -d $_installdir ] ; then 
    runcmd "rm -rf $_installdir"
  fi
  runcmd "make install"
  if [ -L $_linkfrom ] ; then 
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>$CMDLOGFILE
}

function buildsilo(){
  _version=4.10.2
  _installdir=$AOMP_SUPP_INSTALL/silo-$_version
  _linkfrom=$AOMP_SUPP/silo
  _builddir=$AOMP_SUPP_BUILD/silo
  if [ -d $_builddir ] ; then 
    runcmd "rm -rf $_builddir"
  fi
  runcmd "mkdir -p $_builddir"
  runcmd "cd $_builddir"
  runcmd "wget https://wci.llnl.gov/sites/wci/files/2021-01/silo-$_version.tgz"
  runcmd "tar -xzf silo-$_version.tgz"
  runcmd "cd silo-$_version"
  runcmd "mkdir -p $_installdir"
  runcmd "./configure --prefix=$_installdir"
  runcmd "make -j8"
  if [ -d $_installdir ] ; then 
    runcmd "rm -rf $_installdir"
  fi
  runcmd "make install"
  if [ -L $_linkfrom ] ; then 
    runcmd "rm $_linkfrom"
  fi
  runcmd "ln -sf $_installdir $_linkfrom"
  echo "# $_linkfrom is now symbolic link to $_installdir " >>$CMDLOGFILE
}
#---------------------------   Main script starts here -----------------------
CMDLOGFILE=$AOMP_SUPP_BUILD/cmdlog
mkdir -p $AOMP_SUPP_BUILD
if [ "$1" == "-h" ] ; then 
  echo help for $0  not yet available
  exit 0
fi
if [ "$1" == "" ] ; then 
  _components="$SUPPLEMENTAL_COMPONENTS"
else
  _components=$@
fi
# save the current directory
curdir=$PWD
for _component in $_components ; do 
  echo 
  echo "--------- BUILDING SUPPLEMENTAL COMPONENT $_component"
  echo 
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
  else
    echo "ERROR:  Invalid component name $_component" >>$CMDLOGFILE
    echo "        Must be a subset of:  $SUPPLEMENTAL_COMPONENTS"
    exit 0
  fi
  _thisdate=`date`
  echo "# DONE: successful build of $_component at $_thisdate " >>$CMDLOGFILE
done
# restore the current directory
cd $curdir
