#!/bin/bash
rm -f $1.stdout.log $1.stderr.log
./$1 2> $1.stderr.log | tee -a $1.stdout.log 
