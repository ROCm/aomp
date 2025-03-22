#!/bin/bash
rm -f $1.stdout.log $1.stderr.log
./$1 2> $1.stderr.raw.log | tee -a $1.stdout.log 
sed -e "s/Kernel '.*'/Kernel 'xxx'/" $1.stderr.raw.log > $1.stderr.log
