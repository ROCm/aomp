#!/bin/bash

if [ -e ~/local/openmpi/lib/libmpi.so ]; then exit; fi;
if [ -e /opt/openmpi-4.1.5/lib/libmpi.so ]; then exit; fi;
if [ -e /opt/openmpi-4.1.4/lib/libmpi.so ]; then exit; fi;
if [ -e /usr/local/openmpi/lib/libmpi.so ]; then exit; fi;
if [ -e /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so ]; then exit; fi;
if [ -e /usr/lib/openmpi/lib/libmpi.so ]; then exit; fi;
if [ -e /usr/local/lib/libmpi.so ]; then exit ;fi;
echo "No mpi found , not good."

if [ -e /usr/bin/python ]; then exit; fi;
echo "/usr/bin/python is missing"
