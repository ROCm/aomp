#!/bin/bash
export EX=./$1
export PATH=$AOMP/bin:$PATH
which gpurun
set -x
gpurun -help
gpurun -topo
gpurun -rocmsmi $EX
gpurun -nm $EX
gpurun -nr $EX
gpurun -l $EX
gpurun -md 1 $EX
gpurun -m $EX
gpurun -dryrun $EX
gpurun -nomask $EX
gpurun -nomask $EX

gpurun -v      $EX
gpurun -vv     $EX
gpurun -vvv    $EX
