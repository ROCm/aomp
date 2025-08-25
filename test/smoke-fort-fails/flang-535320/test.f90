module foo

contains
   subroutine rout(ids, nb)
       implicit none
       !$omp declare target
       integer(kind=4), dimension(:), intent(in) :: ids
       integer(kind=4), intent(in) :: nb
       integer(kind=4), dimension(1) :: tmp
       tmp = findloc(ids, -nb)
   end subroutine rout
end module

program finder
   implicit none

   !$omp target
   !$omp end target

end program
