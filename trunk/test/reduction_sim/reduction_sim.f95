program main
   use, intrinsic ::  iso_c_binding
   implicit none

   interface 
      subroutine kmpc_rfun_sum_i(a1,a2)  bind(c,name="__kmpc_rfun_sum_i")
         use, intrinsic :: iso_c_binding
         implicit none
         type(c_ptr), value :: a1
         integer(c_int), value :: a2
      end subroutine kmpc_rfun_sum_i
      subroutine kmpc_rfun_sum_lds_i(a1,a2)  bind(c,name="__kmpc_rfun_sum_lds_i")
         use, intrinsic :: iso_c_binding
         implicit none
         type(c_ptr), value :: a1
         integer(c_int), value :: a2
      end subroutine kmpc_rfun_sum_lds_i
      subroutine kmpc_xteamr_i_4x64(val0, r_ptr, tvals, td, &
                      f1, f2, rnv, k, nteams) bind(c,name="__kmpc_xteamr_i_4x64") 
         use, intrinsic :: iso_c_binding
         implicit none
         integer,value :: val0 !  thread local sotrage
         type(c_ptr),value :: r_ptr
         integer(c_int) :: tvals(*)
         type(c_ptr),value :: td 
         type(c_funptr),value :: f1
         type(c_funptr),value  :: f2
         integer,value :: rnv
         integer(8),value ::  k
         integer,value :: nteams
      end subroutine kmpc_xteamr_i_4x64

      subroutine kmpc_rfun_sum_d(a1,a2)  bind(c,name="__kmpc_rfun_sum_d")
         use, intrinsic :: iso_c_binding
         implicit none
         type(c_ptr), value :: a1
         real(8), value :: a2
      end subroutine kmpc_rfun_sum_d
      subroutine kmpc_xteamr_d_4x64(val0, r_ptr, tvals, td, &
                      f1, f2, rnv, k, nteams) bind(c,name="__kmpc_xteamr_d_4x64") 
         use, intrinsic :: iso_c_binding
         implicit none
         real(8),value :: val0 !  thread local sotrage
         type(c_ptr),value :: r_ptr
         real(8) :: tvals(*)
         type(c_ptr),value :: td 
         type(c_funptr),value :: f1
         type(c_funptr),value  :: f2
         real(8),value :: rnv
         integer(8),value ::  k
         integer,value :: nteams
      end subroutine kmpc_xteamr_d_4x64
   end interface

   integer :: nteams = 60  ! should be number of CUs

   ! variables to test reduction of type integer
   integer,target :: a_i = 0
   integer,target :: tvals_i(60)
   integer :: validate_i
   integer :: val0_i

   ! variables to test reduction of type double
   real(8),target :: a_d = 0.0D0
   real(8),target :: tvals_d(60)
   real(8):: validate_d
   real(8) :: val0_d

   ! Control values 
   integer,target :: teams_done
   integer :: idx_bj , k_idx
   integer :: sz = 19200
   integer :: nthreads = 256
   integer(8) :: kc_idx
   teams_done = 0

   print *, "Starting target region "

   !  Simulate the following OpenMP reduction using xteamr ASO helper functions.
   !  !$omp target teams distribute parallel do map(tofrom:a_i) reduction(+:a_i)
   !  do i = 1, sz 
   !     a_i = a_i + i
   !  end do
   !  !$omp end target teams distribute parallel do

   ! Simulate above reduction without reduction clause by using nteams*nthreads
   ! threads. Each thread reduces to the thread local (tl) value val0_i using
   ! the big jump loop.  These tl values are passed to the xteamr function
   ! which reduces within each warp, then across all warps in the team, and
   ! then across teams. The last phase (across teams) reduction is done by the
   ! last team to finish reducing across its warps using the tvals_i array.

   !$omp target teams distribute parallel do &
   !$omp&  map(tofrom: a_i,teams_done) map(alloc:tvals_i) &
   !$omp&  thread_limit(nthreads) num_teams(nteams) &
   !$omp&  firstprivate(sz, nteams, nthreads) private(val0_i, kc_idx, idx_bj) 
   do k_idx = 1, nteams*nthreads
     kc_idx = k_idx - 1
     val0_i = 0
     do idx_bj = k_idx, sz , nteams*nthreads  ! big jump loop
        if( idx_bj .le. sz ) then
           val0_i = val0_i + idx_bj
        endif 
     end do
!    call xteam function to reduce within warp, across warps, and across teams
     CALL kmpc_xteamr_i_4x64(val0_i, c_loc(a_i), tvals_i, c_loc(teams_done),   &
        c_funloc(kmpc_rfun_sum_i), c_funloc(kmpc_rfun_sum_lds_i),              &
        0, kc_idx, nteams)
   end do
   !$omp end target teams distribute parallel do
   print *, "End target region "

   validate_i = ( ( sz + 1 ) * sz ) / 2
   if (a_i .ne. validate_i) then
     print *, "Failed a=" , a_i , " validate=", validate_i
     stop 2
   endif
   print *, "Integer test passed a=", a_i

end program main
