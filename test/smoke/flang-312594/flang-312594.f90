module test_mod
   abstract interface
      function mm_func(ownedIndices) result(iErr)
#ifdef INT_PTR
         integer, dimension(:), pointer :: ownedIndices
#else
         integer, dimension(:) :: ownedIndices
#endif
         integer :: iErr
      end function
   end interface

   contains

end module test_mod

module test
   use test_mod

   implicit none

   contains

   subroutine test_register_method(decompFunc)!{{{
      procedure (mm_func), pointer :: decompFunc

   end subroutine test_register_method!}}}
end module test


program repro
   use test_mod
   call test()
   print *, "Compilation test"
end program repro
