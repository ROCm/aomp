#!/bin/bash

if [ -e /usr/bin/python ]; then
  echo "/usr/bin/python OK"
else
  echo "/usr/bin/python is missing FAIL"
fi
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
else
  echo "No mpi found , not good. FAIL"
fi
