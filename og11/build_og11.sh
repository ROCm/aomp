#!/bin/bash
#
#  build_og11.sh: script to configure and build gcc for amdgcn
#
#  This runs a long time so run this script with nohup like this:
#    nohup build_og11.sh
#  or 
#   ./build_og11.sh 2>&1 | tee build.out
#
#  Script follows some instruction given at https://gcc.gnu.org/wiki/Offloading
#  Written by Greg Rodgers

# Set number of make jobs to speed this up. 
make_jobs=12

# This MAINDIR will contain multiple git repositories clones
MAINDIR=${MAINDIR:-/home/grodgers/git/og11}

# This script requires ROCMLLVM.
ROCMLLVM=${ROCMLLVM:-/opt/rocm/llvm}
if [[ ! -f "${ROCMLLVM}/bin/llvm-mc" ]] ; then 
   echo "ERROR:  missing ${ROCMLLVM}/bin/llvm-mc check LLVM installation"
   exit 1
fi

echo " ============================ OG11BSTEP: Initialize  ================="
mkdir -p $MAINDIR
if [ $? != 0 ] ; then 
   echo "ERROR: No update access to $MAINDIR"
   exit 1
fi
gccmaindir=$MAINDIR/gcc
if [ -d $gccmaindir ] ; then 
   # Get updates
   cd $gccmaindir
   git pull
   git status
else
   # get a new clone of the source 
   echo cd $MAINDIR
   cd $MAINDIR
   echo git clone git://gcc.gnu.org/git/gcc.git
   git clone git://gcc.gnu.org/git/gcc.git
   cd $gccmaindir
   git checkout devel/omp/gcc-11
   git pull
fi

if [ -d $MAINDIR/newlib-cygwin ] ; then 
   # Get updates if any
   echo cd $MAINDIR/newlib-cygwin
   cd $MAINDIR/newlib-cygwin
   echo git pull
   git pull
else
   cd $MAINDIR
   echo git clone git://sourceware.org/git/newlib-cygwin.git
   git clone git://sourceware.org/git/newlib-cygwin.git
fi
[ ! -L $gccmaindir/newlib ] && ln -sf $MAINDIR/newlib-cygwin/newlib $gccmaindir/newlib

# wget tarballs for other components
cd $gccmaindir
if [ ! -L isl ] ; then 
   echo -- getting isl
   wget https://gcc.gnu.org/pub/gcc/infrastructure/isl-0.24.tar.bz2
   bunzip2 isl-0.24.tar.bz2
   tar -xf isl-0.24.tar
   ln -sf isl-0.24 isl
fi
if [ ! -L gmp ] ; then 
   echo -- getting gmp
   wget https://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.2.1.tar.bz2
   bunzip2 gmp-6.2.1.tar.bz2
   tar -xf gmp-6.2.1.tar
   ln -sf gmp-6.2.1 gmp
fi
if [ ! -L mpc ] ; then 
   echo -- getting mpc
   wget https://gcc.gnu.org/pub/gcc/infrastructure/mpc-1.2.1.tar.gz
   gunzip  mpc-1.2.1.tar.gz
   tar -xf mpc-1.2.1.tar
   ln -sf mpc-1.2.1 mpc 
fi
if [ ! -L mpfr ] ; then 
   echo -- getting mpfr
   wget https://gcc.gnu.org/pub/gcc/infrastructure/mpfr-4.1.0.tar.bz2
   bunzip2 mpfr-4.1.0.tar.bz2
   tar -xf mpfr-4.1.0.tar
   ln -sf mpfr-4.1.0 mpfr
fi

installdir="$MAINDIR/install"
[ -d $installdir ] && rm -rf $installdir
mkdir -p $installdir/amdgcn-amdhsa/bin
rsync -av $ROCMLLVM/bin/llvm-ar $installdir/amdgcn-amdhsa/bin/ar
rsync -av $ROCMLLVM/bin/llvm-ar $installdir/amdgcn-amdhsa/bin/ranlib
rsync -av $ROCMLLVM/bin/llvm-mc $installdir/amdgcn-amdhsa/bin/as
rsync -av $ROCMLLVM/bin/llvm-nm $installdir/amdgcn-amdhsa/bin/nm
rsync -av $ROCMLLVM/bin/lld     $installdir/amdgcn-amdhsa/bin/ld
echo " ============================ OG11BSTEP: Configure device ================="
cd $MAINDIR
#  Uncomment next line to start build from scratch
[ -d ./build-amdgcn ] && rm -rf build-amdgcn
mkdir -p build-amdgcn
echo cd build-amdgcn
cd build-amdgcn
WITHOPTS="--with-gmp --with-mpfr --with-mpc "
echo "../gcc/configure --prefix=$installdir -v --target=amdgcn-amdhsa --enable-languages=c,lto,fortran --disable-sjlj-exceptions --with-newlib --enable-as-accelerator-for=x86_64-pc-linux-gnu --with-build-time-tools=$installdir/amdgcn-amdhsa/bin --disable-libquadmath "  | tee ../amdgcnconfig.stdout
../gcc/configure --prefix=$installdir -v --target=amdgcn-amdhsa --enable-languages=c,lto,fortran --disable-sjlj-exceptions --with-newlib --enable-as-accelerator-for=x86_64-pc-linux-gnu --with-build-time-tools=$installdir/amdgcn-amdhsa/bin --disable-libquadmath  2>&1 | tee ../amdgcnconfig.stdout
if [ $? != 0 ] ; then 
   echo "ERROR  configure amdgcn compiler failed"
   exit 1
fi
echo " ============================ OG11BSTEP: device make ================="
make -j$make_jobs
if [ $? != 0 ] ; then 
   echo "ERROR  make amdgcn compiler failed"
   exit 1
fi
echo " ============================ OG11BSTEP: device make install ================="
make install 
echo "Removing symlink for newlib rm $gccmaindir/newlib"
rm $gccmaindir/newlib

cd $MAINDIR
#  Uncomment next line to start build from scratch
[ -d ./build-host ] && rm -rf build-host
echo " ============================ OG11BSTEP: host configure ================="
mkdir -p build-host
echo cd build-host
cd build-host
echo "../gcc/configure --prefix=$installdir -v --build=x86_64-pc-linux-gnu --host=x86_64-pc-linux-gnu --target=x86_64-pc-linux-gnu --enable-offload-targets=amdgcn-amdhsa=$installdir/amdgcn-amdhsa "
../gcc/configure --prefix=$installdir -v --build=x86_64-pc-linux-gnu --host=x86_64-pc-linux-gnu --target=x86_64-pc-linux-gnu --enable-offload-targets=amdgcn-amdhsa=$installdir/amdgcn-amdhsa 2>&1 | tee ../hostconfig.stdout
if [ $? != 0 ] ; then 
   echo "ERROR configure host compiler failed"
   exit 1 
fi
echo " ============================ OG11BSTEP: host make ================="
make -j$make_jobs
if [ $? != 0 ] ; then 
   echo "ERROR  make host compiler failed"
   exit 1
fi
echo " ============================ OG11BSTEP: host install ================="
make install 
echo " ============================ OG11BSTEP: DONE ALL STEPS ================="
