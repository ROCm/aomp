#/bin/bash
#
#  build-deb-aomp.sh:  Using debhelper and alien, build the aomp debian and rpm packages
#                      This script is tested on amd64 and ppc64le linux architectures. 
#                      This script needs packages: debuitils,devscripts,cli-common-dev,alien
#                      A complete build of aomp in /usr/lib/aomp_$AOMP_VERSION_STRING
#                      is required before running this script.
#                      You must also have sudo access to build the debs.
#                      You will be prompted for your password for sudo. 
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

. /etc/lsb-release
RELSTRING=`echo $DISTRIB_ID$DISTRIB_RELEASE | tr -d "."`
echo RELSTRING=$RELSTRING
pkgname=aomp
dirname="aomp_${AOMP_VERSION_STRING}"
sourcedir="/usr/lib/$dirname"
installdir="/usr/lib/$dirname"

DEBFULLNAME="Greg Rodgers"
DEBEMAIL="Gregory.Rodgers@amd.com"
export DEBFULLNAME DEBEMAIL

DEBARCH=`uname -m`
RPMARCH=$DEBARCH
if [ "$DEBARCH" == "x86_64" ] ; then 
   DEBARCH="amd64"
fi
if [ "$DEBARCH" == "ppc64le" ] ; then 
   DEBARCH="ppc64el"
fi
if [ "$DEBARCH" == "aarch64" ] ; then
   DEBARCH="arm64"
fi

tmpdir=/tmp/$USER/build-deb
builddir=$tmpdir/$pkgname
if [ -d $builddir ] ; then 
   echo 
   echo "--- CLEANUP LAST BUILD: rm -rf $builddir"
   sudo rm -rf $builddir
fi

debdir=$PWD/debian
froot=$builddir/$pkgname-$AOMP_VERSION
mkdir -p $froot$installdir
mkdir -p $froot/usr/share/doc/$dirname
echo 
echo "--- PREPARING fake root directory $froot"
rsync -a $sourcedir"/" --exclude "\.git" $froot$installdir
cp $debdir/copyright $froot/usr/share/doc/$dirname/.
cp $debdir/LICENSE.TXT $froot/usr/share/doc/$dirname/.
cp -rp $debdir $froot/debian
if [ "$DEBARCH" != "amd64" ] ; then 
  echo    sed -i -e"s/amd64/$DEBARCH/" $froot/debian/control
  sed -i -e"s/amd64/$DEBARCH/" $froot/debian/control
fi

# Make the install and links file version specific
echo "usr/lib/$dirname usr/lib/aomp" > $froot/debian/$pkgname.links
echo "usr/lib/$dirname/bin/aompExtractRegion /usr/bin/aompExtractRegion" >> $froot/debian/$pkgname.links
echo "usr/lib/$dirname/bin/cloc.sh /usr/bin/cloc.sh" >> $froot/debian/$pkgname.links
echo "usr/lib/$dirname/bin/mymcpu  /usr/bin/mymcpu" >> $froot/debian/$pkgname.links
echo "usr/lib/$dirname/bin/mygpu  /usr/bin/mygpu" >> $froot/debian/$pkgname.links
echo "usr/lib/$dirname/bin/aompversion /usr/bin/aompversion" >> $froot/debian/$pkgname.links
echo "usr/lib/$dirname/bin/aompcc /usr/bin/aompcc" >> $froot/debian/$pkgname.links
echo "usr/lib/$dirname/bin/gpurun /usr/bin/gpurun" >> $froot/debian/$pkgname.links
echo "usr/lib/$dirname" > $froot/debian/$pkgname.install
echo "usr/share/doc/$dirname" >> $froot/debian/$pkgname.install

echo 
echo "--- BUILDING SOURCE TARBALL $builddir/${pkgname}_$AOMP_VERSION.orig.tar.gz "
echo "    FROM FAKEROOT: $froot "
cd $builddir
tar -czf $builddir/${pkgname}_$AOMP_VERSION.orig.tar.gz ${pkgname}-$AOMP_VERSION
echo "    DONE BUILDING TARBALL"

echo 
echo "--- RUNNING dch TO MANANGE CHANGELOG "
cd  $froot
# Skip changelog editor for docker release builds.
echo dch -v ${AOMP_VERSION_STRING} -e --package $pkgname
if [ "$DOCKER" -eq "1" ]; then
  dch -v ${AOMP_VERSION_STRING} --package $pkgname ""
else
  dch -v ${AOMP_VERSION_STRING} --package $pkgname
fi
# Backup the debian changelog to git repo, be sure to commit this 
cp -p $froot/debian/changelog $debdir/.

debuildargs="-us -uc -rsudo --lintian-opts -X files,changelog-file,fields"
echo 
echo "--- BUILDING DEB: debuild $debuildargs "
debuild $debuildargs 
dbrc=$?
echo "    DONE BUILDING DEB WITH RC: $dbrc"
if [ "$dbrc" != "0" ] ; then 
   echo "ERROR during debuild"
   exit $dbrc
fi
# Backup debhelper log to git repo , be sure to commit this
cp -p $froot/debian/${pkgname}.debhelper.log $debdir/.

mkdir -p $tmpdir/debs
if [ -f $tmpdir/debs/${pkgname}_${AOMP_VERSION_STRING}_$DEBARCH.deb ] ; then 
  rm -f $tmpdir/debs/${pkgname}_${AOMP_VERSION_STRING}_$DEBARCH.deb 
fi
# Insert the os release name
cp -p $builddir/${pkgname}_${AOMP_VERSION_STRING}_$DEBARCH.deb $tmpdir/debs/${pkgname}_${RELSTRING}_${AOMP_VERSION_STRING}_$DEBARCH.deb
mkdir -p $tmpdir/rpms
cd $tmpdir/rpms
echo 
echo "--- BUILDING RPM: alien -k --scripts --to-rpm $tmpdir/debs/${pkgname}_${AOMP_VERSION_STRING}_$DEBARCH.deb"
sudo alien -k --scripts --to-rpm $builddir/${pkgname}_${AOMP_VERSION_STRING}_$DEBARCH.deb 

echo 
echo "DONE Debian package is in $tmpdir/debs/${pkgname}_${AOMP_VERSION_STRING}_$DEBARCH.deb"
echo "     rpm package is in $tmpdir/rpms/${pkgname}_${AOMP_VERSION_STRING}.$RPMARCH.rpm"
echo 
