Expected output:

             none_shared    21
            none_private    21
       none_firstprivate    21
          private_shared    21
    private_firstprivate    21
     firstprivate_shared    21
    firstprivate_private    21

-----------------------------------------------------------------------------
Semantic check fails on test

./shared_crayptr.f90:24:14: error: The DEFAULT(NONE) clause requires that 'var' must be listed in a data-sharing attribute clause
      var(1) = var(1) / 2
               ^^^
./shared_crayptr.f90:25:38: error: The DEFAULT(NONE) clause requires that 'var' must be listed in a data-sharing attribute clause
      print '(A24,I6)', 'none_shared', var(1)
                                       ^^^
make: *** [Makefile:13: shared_crayptr] Error 1


