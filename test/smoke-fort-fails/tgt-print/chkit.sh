#!/bin/bash
diff -w $1.stderr.log chk.stderr
diff -w $1.stdout.log chk.stdout
