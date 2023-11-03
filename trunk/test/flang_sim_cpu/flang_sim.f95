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

   ! variables to test reduction of type integer
   integer,target :: a_i = 0
   integer,target :: team_vals_i(120)
   integer :: validate_i
   integer :: val0_i

   ! variables to test reduction of type double
   real(8),target :: a_d = 0.0D0
   real(8),target :: team_vals_d(120)
   real(8):: validate_d
   real(8) :: val0_d

   ! Control values 
   integer,target :: teams_done
   integer :: i, k
   integer :: size = 19200
   integer :: nteams = 60
   integer :: nthreads = 256
   integer(8) :: cidx
   teams_done = nteams

   !!$omp target teams distribute parallel do map(tofrom: a_i) num_teams(nteams) thread_limit(nthreads)
   do k= 1, nteams*nthreads
     cidx = k-1
     val0_i = 0 ! this will need to be thread local in a target region 
     do i = k, size , nteams*nthreads
        if( i .le. size ) then
           val0_i = val0_i + i
        endif 
     end do
     !  call the cross team helper function 
     CALL kmpc_xteamr_i_4x64(val0_i, c_loc(a_i), team_vals_i, c_loc(teams_done), &
        c_funloc(kmpc_rfun_sum_i), c_funloc(kmpc_rfun_sum_i), 0, cidx, nteams)
   end do
   !!$omp end target teams distribute parallel do

   validate_i = ( ( size + 1 ) * size ) / 2
   if (a_i .ne. validate_i) then
     print *, "Failed a=" , a_i , " validate=", validate_i
     stop 2
   endif
   print *, "Integer test passed a=", a_i

   !! Test double 
   teams_done = nteams
   !!$omp target teams distribute parallel do map(tofrom: a_d) num_teams(nteams) thread_limit(nthreads)
   do k= 1, nteams*nthreads
     cidx = k-1
     val0_d = 0.0D0 ! this will need to be thread local in a target region 
     do i = k, size , nteams*nthreads
        if( i .le. size ) then
           val0_d = val0_d + i
        endif 
     end do
     !  call the cross team helper function 
     CALL kmpc_xteamr_d_4x64(val0_d, c_loc(a_d), team_vals_d, c_loc(teams_done), &
        c_funloc(kmpc_rfun_sum_d), c_funloc(kmpc_rfun_sum_d), 0.0D0, cidx, nteams)
   end do
   !!$omp end target teams distribute parallel do

   validate_d = ( ( size + 1 ) * size ) / 2
   if (a_d .ne. validate_d) then
     print *, "Failed a=" , a_d , " validate=", validate_d
     stop 2
   endif
   print *, "real(8) test passed a=", a_d

end program main
