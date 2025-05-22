#!/bin/bash
flterr=$1.flterr.log
cat $1.stderr.log | sed -e 's/0x[0-9a-f]*/0xXXXX/ig' > $flterr
diff -w $flterr       chk.stderr
diff -w $1.stdout.log chk.stdout
