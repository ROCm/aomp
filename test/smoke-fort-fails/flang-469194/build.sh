#!/bin/bash
export PATH=${AOMP}/bin:${PATH}

echo
which flang-new

CPPFLAGS="-DLINUX -DLITTLE_ENDIAN -DLITTLE -DADDRESS64 -DPARKIND1_SINGLE -DSFX_ARO -DSFX_ASC -DSFX_OL -DSFX_TXT -DSFX_FA -DSFX_LFI -DARO -DOL -DASC -DTXT -DFA -DLFI"
CFLAGS="-fPIC -fopenmp -fconvert=big-endian -cpp -ffree-form -march=znver4 -ffp-contract=fast -O3"

echo flang-new ${CPPFLAGS} ${CFLAGS} -c ./mode_snow3l.F90 -o ./mode_snow3l.o
flang-new ${CPPFLAGS} ${CFLAGS} -c ./mode_snow3l.F90 -o ./mode_snow3l.o

echo flang-new ${CPPFLAGS} ${CFLAGS} -c ./init_snow_lw.F90 -o ./init_snow_lw.o
flang-new ${CPPFLAGS} ${CFLAGS} -c ./init_snow_lw.F90 -o ./init_snow_lw.o
