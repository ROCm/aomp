#!/bin/bash
rm -f chk src dst dst1 dst2 dst3
echo hello > chk
cp chk src
./rename-func
