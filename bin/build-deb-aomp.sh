#/bin/bash
#
#  build-deb-aomp.sh:  Using debhelper and alien, build the aomp debian and rpm packages
#                      This script is tested on amd64 and ppc64le linux architectures. 
#                      This script needs packages: debuitils,devscripts,cli-common-dev,alien
#                      A complete build of aomp in /opt/rocm/aomp_$AOMP_VERSION_STRING
#                      is required before running this script.
#                      You must also have sudo access to build the debs.
#                      You will be prompted for your password for sudo. 
#
pkgname=aomp
# FIXME: Get version and mod from AOMP_VERSION_STRING file in directory with this script
version="0.6"
mod="0"
dirname="aomp_$version-$mod"
sourcedir="/opt/rocm/$dirname"
installdir="/opt/rocm/$dirname"

DEBFULLNAME="Greg Rodgers"
DEBEMAIL="Gregory.Rodgers@amd.com"
export DEBFULLNAME DEBEMAIL

DEBARCH=`uname -p`
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
froot=$builddir/$pkgname-${version}
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
echo "opt/rocm/$dirname opt/rocm/aomp" > $froot/debian/$pkgname.links
echo "opt/rocm/$dirname/bin/bundle.sh  /usr/bin/bundle.sh" >> $froot/debian/$pkgname.links
echo "opt/rocm/$dirname/bin/unbundle.sh  /usr/bin/unbundle.sh" >> $froot/debian/$pkgname.links
echo "opt/rocm/$dirname/bin/cloc.sh /usr/bin/cloc.sh" >> $froot/debian/$pkgname.links
echo "opt/rocm/$dirname/bin/mymcpu  /usr/bin/mymcpu" >> $froot/debian/$pkgname.links
echo "opt/rocm/$dirname/bin/mygpu  /usr/bin/mygpu" >> $froot/debian/$pkgname.links
echo "opt/rocm/$dirname/bin/aompversion /usr/bin/aompversion" >> $froot/debian/$pkgname.links
echo "opt/rocm/$dirname" > $froot/debian/$pkgname.install
echo "usr/share/doc/$dirname" >> $froot/debian/$pkgname.install

echo 
echo "--- BUILDING SOURCE TARBALL $builddir/${pkgname}_${version}.orig.tar.gz "
echo "    FROM FAKEROOT: $froot "
cd $builddir
tar -czf $builddir/${pkgname}_${version}.orig.tar.gz ${pkgname}-${version}
echo "    DONE BUILDING TARBALL"

echo 
echo "--- RUNNING dch TO MANANGE CHANGELOG "
cd  $froot
echo dch -v ${version}-${mod} -e --package $pkgname
dch -v ${version}-${mod} --package $pkgname
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
if [ -f $tmpdir/debs/${pkgname}_$version-${mod}_$DEBARCH.deb ] ; then 
  rm -f $tmpdir/debs/${pkgname}_$version-${mod}_$DEBARCH.deb 
fi
mv $builddir/${pkgname}_$version-${mod}_$DEBARCH.deb $tmpdir/debs/.
mkdir -p $tmpdir/rpms
cd $tmpdir/rpms
echo 
echo "--- BUILDING RPM: alien -k --scripts --to-rpm $tmpdir/debs/${pkgname}_$version-${mod}_$DEBARCH.deb"
sudo alien -k --scripts --to-rpm $tmpdir/debs/${pkgname}_$version-${mod}_$DEBARCH.deb 

echo 
echo "DONE Debian package is in $tmpdir/debs/${pkgname}_$version-${mod}_$DEBARCH.deb"
echo "     rpm package is in $tmpdir/rpms/${pkgname}_$version-${mod}.$RPMARCH.rpm"
echo 
