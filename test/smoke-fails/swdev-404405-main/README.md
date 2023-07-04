# README

This reproducer demonstrates poor performance in a Fortran OpenMP program compiled with flang due to spurious data maps when using unified shared memory.

Use the provided `Makefile` to compile and modify it accordingly.

## Compiler version
```
flang --version
AOMP_STANDALONE_17.0-2 flang-classic version 17.0.0 (ssh://lstringe@gerrit-git.amd.com:29418/lightning/ec/llvm-project a7cef3f8eedc756e26ae28c77b7b045042ab1766)
Target: x86_64-unknown-linux-gnu
Thread model: posix
```

## Expected output
As of June 6,  2023, we see the following performance on MI300A (sh5)
```
./reproducer.x
 Data size (read and write):    1.760000000000000      GB
 System allocator
 Elapsed:   0.3891510000000000       s Bandwidth:    4.522666008824338
  GB/s
 Elapsed:   9.0490000000000001E-002  s Bandwidth:    19.44966294618190
  GB/s
 Elapsed:   9.2017000000000002E-002  s Bandwidth:    19.12690046404469
  GB/s
 Elapsed:   9.4160999999999995E-002  s Bandwidth:    18.69139027835303
  GB/s
 Elapsed:   9.2069999999999999E-002  s Bandwidth:    19.11589008363202
  GB/s
 Elapsed:   9.4388000000000000E-002  s Bandwidth:    18.64643810653897
  GB/s
 Elapsed:   9.2088000000000003E-002  s Bandwidth:    19.11215359221614
  GB/s
 Elapsed:   9.4091999999999995E-002  s Bandwidth:    18.70509713897037
  GB/s
 Elapsed:   9.2099000000000000E-002  s Bandwidth:    19.10987089979261
  GB/s
 Elapsed:   9.3996999999999997E-002  s Bandwidth:    18.72400182984563
  GB/s
 Hipmalloc
 Elapsed:   8.8668999999999998E-002  s Bandwidth:    19.84910171536839
  GB/s
 Elapsed:   8.8756000000000002E-002  s Bandwidth:    19.82964531975303
  GB/s
 Elapsed:   8.9680999999999997E-002  s Bandwidth:    19.62511568782685
  GB/s
 Elapsed:   8.8456000000000007E-002  s Bandwidth:    19.89689789273763
  GB/s
 Elapsed:   8.9788999999999994E-002  s Bandwidth:    19.60151020726370
  GB/s
 Elapsed:   8.8735999999999995E-002  s Bandwidth:    19.83411467724486
  GB/s
 Elapsed:   9.0055999999999997E-002  s Bandwidth:    19.54339522075153
  GB/s
 Elapsed:   8.8539999999999994E-002  s Bandwidth:    19.87802123334086
  GB/s
 Elapsed:   8.9422000000000001E-002  s Bandwidth:    19.68195746013285
  GB/s
 Elapsed:   8.8933999999999999E-002  s Bandwidth:    19.78995659702701
  GB/s
 omp_target_alloc
 Elapsed:   9.2007000000000005E-002  s Bandwidth:    19.12897931679111
  GB/s
 Elapsed:   8.8537000000000005E-002  s Bandwidth:    19.87869478297209
  GB/s
 Elapsed:   8.8371000000000005E-002  s Bandwidth:    19.91603580360073
  GB/s
 Elapsed:   8.8528999999999997E-002  s Bandwidth:    19.88049113849699
  GB/s
 Elapsed:   8.8590000000000002E-002  s Bandwidth:    19.86680212213568
  GB/s
 Elapsed:   8.8291999999999995E-002  s Bandwidth:    19.93385584197889
  GB/s
 Elapsed:   8.8662000000000005E-002  s Bandwidth:    19.85066883219417
  GB/s
 Elapsed:   8.8589000000000001E-002  s Bandwidth:    19.86702638025037
  GB/s
 Elapsed:   8.8764999999999997E-002  s Bandwidth:    19.82763476595505
  GB/s
 Elapsed:   8.8540999999999995E-002  s Bandwidth:    19.87779672694006
  GB/s
 Elapsed:   8.9824000000000001E-002  s Bandwidth:    19.59387246170289
  GB/s
 Elapsed:   8.8496000000000005E-002  s Bandwidth:    19.88790453805822
  GB/s
 Elapsed:   8.8810000000000000E-002  s Bandwidth:    19.81758810944713
  GB/s
 Elapsed:   8.8346999999999995E-002  s Bandwidth:    19.92144611588396
  GB/s
 Elapsed:   8.8463000000000000E-002  s Bandwidth:    19.89532346856878
  GB/s
 Elapsed:   8.8957999999999995E-002  s Bandwidth:    19.78461745992491
  GB/s
 Elapsed:   8.8543999999999998E-002  s Bandwidth:    19.87712323816408
  GB/s
 Elapsed:   8.8303999999999994E-002  s Bandwidth:    19.93114694691067
  GB/s
 Elapsed:   8.8695999999999997E-002  s Bandwidth:    19.84305943898259
  GB/s
 Elapsed:   8.8364999999999999E-002  s Bandwidth:    19.91738810615063
  GB/s
```
