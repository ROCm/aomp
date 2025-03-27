-----------------------------------------------------------------------------
Semantic check fails on test

$ AOMP=/COD/LATEST/aomp/llvm make run
/COD/LATEST/aomp/llvm/bin/flang-new    -fopenmp  -D__OFFLOAD_ARCH_gfx90a__ crayptr-default.f90 -o crayptr-default
error: Semantic errors in crayptr-default.f90
./crayptr-default.f90:23:8: error: The DEFAULT(NONE) clause requires that 'var' must be listed in a data-sharing attribute clause
      n2=var(2,n)
         ^^^
make: *** [../Makefile.rules:62: crayptr-default] Error 1


-----------------------------------------------------------------------------
If default(none) is removed:

$ AOMP=/COD/LATEST/aomp/llvm make run
/COD/LATEST/aomp/llvm/bin/flang-new    -fopenmp  -D__OFFLOAD_ARCH_gfx90a__ crayptr-default.f90 -o crayptr-default
./crayptr-default  2>&1 | tee run.log
 101 202 303 404 505 606 707 808
 Success
echo > /dev/null 2>&1 | tee -a run.log

-----------------------------------------------------------------------------
Adding var to shared (correctly) reports:

$ AOMP=/COD/LATEST/aomp/llvm make run
/COD/LATEST/aomp/llvm/bin/flang-new    -fopenmp  -D__OFFLOAD_ARCH_gfx90a__ crayptr-default.f90 -o crayptr-default
error: Semantic errors in crayptr-default.f90
./crayptr-default.f90:19:34: error: Cray Pointee 'var' may not appear in SHARED clause, use Cray Pointer 'ivar' instead
  !$omp& shared (ivar,npair,result,var)
                                   ^^^
make: *** [../Makefile.rules:62: crayptr-default] Error 1

-----------------------------------------------------------------------------
Removing ivar from shared (correctly) reports:

$ AOMP=/COD/LATEST/aomp/llvm make run
/COD/LATEST/aomp/llvm/bin/flang-new    -fopenmp  -D__OFFLOAD_ARCH_gfx90a__ crayptr-default.f90 -o crayptr-default
error: Semantic errors in crayptr-default.f90
./crayptr-default.f90:22:8: error: The DEFAULT(NONE) clause requires that the Cray Pointer 'ivar' must be listed in a data-sharing attribute clause
      n1=var(1,n)
         ^^^
make: *** [../Makefile.rules:62: crayptr-default] Error 1

