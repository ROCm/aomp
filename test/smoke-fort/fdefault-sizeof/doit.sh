#!/bin/bash

SRC=sizeofdouble.f90
EXE=sizeofdouble
FLANG=${FLANG:-flang-new}

FC=$AOMP/bin/$FLANG
FFLAGS=-fdefault-real-8

declare -a flags=(
        ""
        "-fdefault-integer-8"
        "                    -fdefault-real-8"
        "                                     -fdefault-double-8"
        "-fdefault-integer-8 -fdefault-real-8"
        "-fdefault-integer-8                  -fdefault-double-8"
        "                    -fdefault-real-8 -fdefault-double-8"
        "-fdefault-integer-8 -fdefault-real-8 -fdefault-double-8"
        )

for f in "${flags[@]}"; do 
    rm -f $EXE
    cmd="$FC $f $SRC -o $EXE"
    echo "-------"
    echo "flags: $f"
    $cmd
    if [ -x $EXE ]; then
        ./$EXE
    fi
done

exit 0
