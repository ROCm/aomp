#!/bin/bash
#
#  package_release_tarball.sh: Build the tarball for aomp release
#
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/aomp_common_vars"
# --- end standard header ----

pkgname=aomp
echo "Building $pkgname package"

dirname="aomp_${AOMP_VERSION_STRING}"
sourcedir="/usr/lib/$dirname"
installdir="/usr/lib/$dirname"

tmpdir="/tmp/build-tar"
builddir="$tmpdir/$pkgname"
froot="$builddir/$pkgname-$AOMP_VERSION"

if [ -d "$builddir" ] ; then
   echo
   echo "--- CLEANUP LAST BUILD: rm -rf $builddir"
   rm -rf "$builddir"
fi

mkdir -p "$froot$installdir"
rsync -a "$sourcedir/" --exclude ".*" "$froot$installdir"

cd "$froot$installdir/../" || exit
tarball="$AOMP_REPOS/../aomp-${AOMP_VERSION_STRING}.tar.gz"
tar -h -czf $tarball $dirname
