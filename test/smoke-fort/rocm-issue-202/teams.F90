program teams
implicit none
integer :: i
!$omp target teams
select case (i)
case(1)
end select
!$omp end target teams
end program
