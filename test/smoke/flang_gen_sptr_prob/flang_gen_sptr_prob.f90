 MODULE buoyancy_mod
    REAL, DIMENSION(:,:,:), ALLOCATABLE   :: sums_l
    INTEGER, DIMENSION(:,:), ALLOCATABLE :: nzb_s_inner
    INTEGER :: nxl, nxr, nys, nyn, nzt 

 CONTAINS
    SUBROUTINE calc_mean_profile( var, pr )
       IMPLICIT NONE
       INTEGER ::  i, j, k, omp_get_thread_num, pr, tn
       REAL, DIMENSION(:,:,:) ::  var
          tn = 0
          sums_l(:,pr,tn) = 0.0

          !$omp target teams distribute parallel do private( i, j, k ) reduction(+:sums_l)
          DO  i = nxl, nxr
             DO  j =  nys, nyn
             	!$omp simd
                DO  k = nzb_s_inner(j,i), nzt+1
                   sums_l(k,pr,tn) = sums_l(k,pr,tn) + var(k,j,i)
                ENDDO
             ENDDO
          ENDDO
          !$omp end target teams distribute parallel do

    END SUBROUTINE calc_mean_profile

 END MODULE buoyancy_mod
 program main
   print *,'hi'
 end program
