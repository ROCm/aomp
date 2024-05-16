#!/bin/bash
#
#  build-rpm.sh: Build the rpm for SLES15 SP4 and Centos 7-9
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

osname=$(cat /etc/os-release | grep -e ^NAME=)
version=$(cat /etc/os-release | grep -e ^VERSION=)
rpmname="Not_Found"
if [[ $osname =~ "Red Hat" ]]; then
  echo "Red Hat found!!!"
  rpmname=${1:-aomp_REDHAT_7}
elif [[ $osname =~ "SLES" ]]; then
  echo "SLES15_SP4 found!!!"
  rpmname=${1:-aomp_SLES15_SP4}
elif [[ $osname =~ "CentOS" ]]; then
  echo "CENTOS found!!!"
  if [[ $version =~ "9" ]]; then
    rpmname=${1:-aomp_CENTOS_9}
  elif [[ $version =~ "8" ]]; then
    rpmname=${1:-aomp_CENTOS_8}
  elif [[ $version =~ "7" ]]; then
    rpmname=${1:-aomp_CENTOS_7}
  fi
fi

echo "rpmname: $rpmname"

# Ensure the rpmbuild tool from rpm-build package is available
rpmbuild_loc=`which rpmbuild 2>/dev/null`
if [ -z "$rpmbuild_loc" ];then
   echo
   echo "ERROR:  You need to install rpm-build for $0"
   echo
   export HOME=$savehome
   cd $curdir
   exit 1
fi

# For the life of this script, change home directory 
curdir=$PWD
tmphome="/tmp/$USER/home"
savehome=$HOME
export HOME=$tmphome

echo --- checking for $tmphome/rpmbuild
if [ -d "$tmphome/rpmbuild" ] ; then 
  echo --- cleanup from previous call to $0
  echo --- rm -rf $tmphome/rpmbuild
  rm -rf $tmphome/rpmbuild
fi

echo --- mkdir -p $tmphome/rpmbuild/SOURCES/$rpmname/usr/lib
mkdir -p $tmphome/rpmbuild/SOURCES/$rpmname/usr/lib
echo --- rsync -a --delete ${AOMP}_${AOMP_VERSION_STRING} ~/rpmbuild/SOURCES/$rpmname/usr/lib
rsync -a --delete ${AOMP}_${AOMP_VERSION_STRING} ~/rpmbuild/SOURCES/$rpmname/usr/lib
echo --- mkdir -p $tmphome/rpmbuild/SPECS
mkdir -p $tmphome/rpmbuild/SPECS

tmpspecfile=$tmphome/rpmbuild/SPECS/$rpmname.spec
echo --- cp $thisdir/$rpmname.spec $tmpspecfile
cp $thisdir/$rpmname.spec $tmpspecfile
echo --- sed -ie "s/__VERSION1/$AOMP_VERSION/" $tmpspecfile
sed -ie "s/__VERSION1/$AOMP_VERSION/" $tmpspecfile
sed -ie "s/__VERSION2_STRING/$AOMP_VERSION_STRING/" $tmpspecfile
sed -ie "s/__VERSION3_MOD/$AOMP_VERSION_MOD/" $tmpspecfile
cat $thisdir/debian/changelog | grep -v " --" | grep -v "UNRELEASED" >>$tmpspecfile

echo --- cd ~/rpmbuild/SOURCES
cd ~/rpmbuild/SOURCES

if [ ! -d ${rpmname} ] ; then
      echo 
      echo "ERROR: Missing directory $tmphome/rpmbuild/SOURCES/${rpmname}"
      echo 
      export HOME=$savehome
      cd $curdir
      exit 1
fi

# tar and zip the package contents 
echo --- rm -f ${rpmname}.tar.gz
rm -f ${rpmname}.tar.gz
echo --- tar zcf ${rpmname}.tar.gz ${rpmname}
tar zcf ${rpmname}.tar.gz ${rpmname}
echo --- ls -l $PWD/${rpmname}.tar.gz
ls -l $PWD/${rpmname}.tar.gz
echo --- done tar

# build the package
echo 
echo ------ STARTING rpmbuild -----
echo 
echo --- rpmbuild -ba ../SPECS/${rpmname}.spec
rpmbuild -ba ../SPECS/${rpmname}.spec
rc=$?
echo ------ DONE rpmbuild rc is $rc -----
if [ $rc != 0 ] ; then 
   echo 
   echo "ERROR during rpmbuild"
   echo 
   export HOME=$savehome
   cd $curdir
   exit 1
fi

outfile="$tmphome/rpmbuild/RPMS/x86_64/${rpmname}-${AOMP_VERSION_STRING}.x86_64.rpm"
echo " --- The output rpm file is $outfile."
echo "--- ls -lh $outfile"
ls -lh $outfile
echo    

export HOME=$savehome
cd $curdir
