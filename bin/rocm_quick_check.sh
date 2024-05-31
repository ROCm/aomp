#!/bin/bash
echo listing /opt 
ls /opt
find /opt/rocm* -name clang

if [ -e /usr/bin/patch  ]; then
  echo "/usr/bin/patch OK"
else
  echo "/usr/bin/patch is missing FAIL"
fi

if [ -e /usr/bin/cmake  ]; then
  echo "/usr/bin/cmake OK"
else
  echo "/usr/bin/cmake is missing FAIL"
fi

if [ -e /usr/bin/python ]; then
  echo "/usr/bin/python OK"
else
  echo "/usr/bin/python is missing FAIL"
fi

echo "Checking for libnvida compute pieces "
if [ -e /usr/bin/dpkg ]; then
  dpkg -l | grep -i nvid
else 
	echo skipping dpkg check
fi
echo "Done checking for libnvidia"

if [ -e ~/local/openmpi/lib/libmpi.so ]; then
  echo "~/local/openmpi/lib/libmpi.so OK"
elif [ -e /opt/openmpi-4.1.5/lib/libmpi.so ]; then
  echo "/opt/openmpi-4.1.5/lib/libmpi.so OK"
elif [ -e /opt/openmpi-4.1.4/lib/libmpi.so ]; then
  echo "/opt/openmpi-4.1.4/lib/libmpi.so  OK"
elif [ -e /usr/local/openmpi/lib/libmpi.so ]; then
  echo "/usr/local/openmpi/lib/libmpi.so OK"
elif [ -e /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so ]; then
  echo "/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so OK"
elif [ -e /usr/lib/openmpi/lib/libmpi.so ]; then
  echo "/usr/lib/openmpi/lib/libmpi.so OK"
elif [ -e /usr/local/lib/libmpi.so ]; then
  echo "/usr/local/lib/libmpi.so OK"
elif [ -e /usr/local/lib64/libmpi.so ]; then
  echo "/usr/local/lib64/libmpi.so OK"
elif [ -e /usr/lib64/openmpi/lib/libmpi.so ]; then
  echo "/usr/lib64/openmpi/lib/libmpi.so OK"
else
  echo "No mpi found , not good. FAIL"
fi


# Look for FileCheck on the system in various places.
# Check local AOMP install first.
SYSFILECHECK=`ls /usr/bin | grep -m1 -e "FileCheck"`
if [ "$SYSFILECHECK" == "" ]; then
	SYSFILECHECK="FileCheck"
fi
AOMP=`ls -d /opt/rocm-*/llvm | head -1`
echo $AOMP
SYSLLVM=`ls /usr/lib | grep -m1 -e "llvm-[0-9]\+"`
if [ -e "$AOMP/bin/FileCheck" ]; then
  echo "$AOMP/bin/FileCheck OK"
elif [ -e /usr/lib/aomp/bin/FileCheck ]; then
  echo "/usr/lib/aomp/bin/FileCheck OK"
elif [ -e $HOME/git/aomp-test/FileCheck ]; then
  echo "$HOME/git/aomp-test/FileCheck OK"
elif [ -e /usr/lib/$SYSLLVM/bin/FileCheck ]; then
  echo "/usr/lib/$SYSLLVM/bin/FileCheck OK"
elif [ -e /usr/bin/$SYSFILECHECK ]; then
  echo "/usr/bin/$SYSFILECHECK OK"
else
  echo "Warning ----Warning---- FileCheck was not found and is needed by smoke tests."
  echo "FileCheck notfound. May need to install llvm-XY-tools (where XY is llvm version)."
fi


echo "Checking for SRIOV "
if [ -e /sbin/lspci ]; then
  /sbin/lspci | grep VMware
else
  lspci | grep VMware
fi
echo "Checking for Hyper"
lscpu |grep Hyper

echo "Done checking for SRIOV"
