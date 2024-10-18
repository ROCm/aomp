#!/bin/bash
rm -f $1.stdout.log $1.stderr.log
./$1 > $1.stdout.log 2> $1.stderr.log
