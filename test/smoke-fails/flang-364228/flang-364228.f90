module array_m

      integer, parameter ::LEN_MAX=128
      !===========================================================
      !# <STRUCTURE NAME=r1_t>
      !# <DESCRIPTION>Pointer for one-dimension real variables.</DESCRIPTION>
      type r1_t
         double precision, allocatable :: vali(:)       !# <VAR NAME=val TYPE=real(:)>Values</VAR>
         character(len = LEN_MAX) :: name          !# <VAR NAME=name TYPE=character>Name of the pointer</VAR>
         type(r1_t), pointer :: next           !# <VAR NAME=next TYPE=r1_t>Next pointer</VAR>
         integer :: dim1
      end type r1_t
      !# </STRUCTURE>
      !===========================================================

      contains
      !=================================================================
      !# <SUBROUTINE NAME=init_r1>
      !# <DESCRIPTION>
      !#    Routine to nullify all the pointers of the structure and
      !#    set the parameters to their default value
      !# </DESCRIPTION>
      !# <VAR NAME=r1 TYPE=r1_t IO=In/Out>r1 pointer to be initialized</VAR>
      !
      subroutine init_r1(r1)

         implicit none

         ! ----------------------
         type(r1_t), pointer :: r1
         ! ----------------------

         r1%name = ""
         r1%dim1=0
         nullify(r1%next)

      end subroutine init_r1
      !# </SUBROUTINE>

      !=================================================================
      !# <SUBROUTINE NAME=find_r1>
      !# <DESCRIPTION>
      !#    Routine to find a pointer in a chained list of pointers from its name
      !# </DESCRIPTION>
      !# <VAR NAME=r1 TYPE=r1_t IO=In>First pointer of the list</VAR>
      !# <VAR NAME=name TYPE=character IO=In>Name of the pointer to find</VAR>
      !# <VAR NAME=res_ptr TYPE=r1_t IO=Out>Found pointer</VAR>
      !# <VAR NAME=resfound TYPE=logical IO=Out OPTIONAL=TRUE>Status of the search: true = found, false = not found</VAR>
      !
      subroutine find_r1(r1,name,res_ptr,resfound)

         implicit none

         ! ----------------------
         type(r1_t), pointer :: r1
         character(len = *), intent(in) :: name
         type(r1_t), pointer :: res_ptr
         logical, intent(out), optional :: resfound
         ! ----------------------
         type(r1_t), pointer :: ar1
         logical :: done,found
         ! ----------------------

         done = .false.
         found = .false.
         if (associated(r1)) then
            ar1 => r1
         else
            nullify(ar1)
            done = .true.
         end if
         do while ((.not.done).and.(.not.found))

            if (ar1%name==name) then
               found = .true.
               res_ptr => ar1
            end if

            if (associated(ar1%next)) then
               ar1 => ar1%next
            else
               done = .true.
            end if
         end do

         if (present(resfound)) then
            resfound = found
         else
            if (.not.found) then
               nullify(res_ptr)
               STOP "in find_r1, pointer not found: "
            end if
         end if

      end subroutine find_r1
      !# </SUBROUTINE>
      !=================================================================

      !=================================================================
      !# <SUBROUTINE NAME=add_new_r1>
      !# <DESCRIPTION>
      !#    Routine to add a pointer at the last position of a chained list of pointers
      !# </DESCRIPTION>
      !# <VAR NAME=r1 TYPE=r1_t IO=In>First pointer of the chained list</VAR>
      !# <VAR NAME=name TYPE=character IO=In OPTIONAL=TRUE>Name of the pointer</VAR>
      !# <VAR NAME=new_ptr TYPE=r1_t IO=Out OPTIONAL=TRUE>New pointer</VAR>
      !
      subroutine add_new_r1(r1,dim1,name,new_ptr)

         implicit none

         ! ----------------------
         type(r1_t), pointer :: r1
         integer, intent(in) :: dim1
         character(len = *), intent(in), optional :: name
         type(r1_t), pointer, optional :: new_ptr
         ! ----------------------
         character(len = LEN_MAX) :: thename
         type(r1_t), pointer :: newr1,lastr1
         integer :: ires
         ! ----------------------

         ! get the name
         if (present(name)) then
            thename = name
         else
            thename = ""
         end if

         ! check
         if (dim1<0) then
            STOP "in add_new_r1: dim1 of "//trim(thename)//" is less than zero "
         end if

         ! find the last non allocated pointer
         if (.not.associated(r1)) then
            allocate(r1, stat=ires)
            if (ires /= 0) then
               STOP "[MEMORY error] Unable to allocate memory in add_new_r1"
            end if
            newr1 => r1
         else
            lastr1 => r1
            do while (associated(lastr1%next))
               lastr1 => lastr1%next
            end do
            allocate(lastr1%next, stat=ires)
            if (ires /= 0) then
               STOP "[MEMORY error] Unable to allocate memory in add_new_r1"
            end if
            newr1 => lastr1%next
         end if

         ! allocate the pointer
         call init_r1(newr1)


         ! allocate the array
         allocate(newr1%vali(dim1), stat=ires)
         if (ires /= 0) then
            STOP "[MEMORY error] In add_new_r1 unable to allocate an array "
         end if
         newr1%vali(1:dim1) = 0.0
         newr1%dim1 = dim1

         ! additional parameters
         if (present(name)) then
            newr1%name = name
         end if
         if (present(new_ptr)) then
            new_ptr => newr1
         end if

      end subroutine add_new_r1
      !# </SUBROUTINE>
      !=================================================================


     !=================================================================
      !# <SUBROUTINE NAME=print_r1>
      !# <DESCRIPTION>
      !#    Routine to print a pointer
      !# </DESCRIPTION>
      !# <VAR NAME=r1 TYPE=r1_t IO=In/Out>r1 pointer to be printed</VAR>
      !
      subroutine print_r1(r1)

         implicit none

         ! ----------------------
         type(r1_t), pointer :: r1
         ! ----------------------
         type(r1_t), pointer :: ar1
         logical :: done
         ! ----------------------

         done = .false.
         if (associated(r1)) then
            ar1 => r1
         else
            done = .true.
         end if
         do while (.not.done)
            write(6,'(a)') "============================="
            write(6,'(a)') "Name: "//trim(ar1%name)
            write(6,'(a,i0)') "Size: ", ar1%dim1
            if (ar1%dim1 .lt. 1) then
               write(6,'(a)')"No data in this arrays"
            else
               write(6,'(a,f14.2,3(a,i0,a,f14.2))')"From val[1]=",ar1%vali(1),", val[",2,"]=",ar1%vali(2),",&
               val[",ar1%dim1-1,"]=",ar1%vali(ar1%dim1-1),", val[",ar1%dim1,"]=",ar1%vali(ar1%dim1)
            endif

            if (associated(ar1%next)) then
               ar1 => ar1%next
            else
               done = .true.
            end if
         end do
         write(6,'(a)') "============================="

      end subroutine print_r1
      !# </SUBROUTINE>
      !=================================================================

end module array_m

module calcule_m
   use array_m
   implicit none

   contains

     !=================================================================
      !# <SUBROUTINE NAME=init_i_r1>
      subroutine init_i_r1(r1)
      implicit none
      type(r1_t), pointer :: r1
      integer ::i
      if (.not. associated(r1)) STOP "cannot initialize non existing r1"
      do i=1,r1%dim1
         r1%vali(i)=i*10.0
      enddo

      end subroutine init_i_r1
      !# </SUBROUTINE>
      !=================================================================

     !=================================================================
      !# <SUBROUTINE NAME=init_val_r1>
      subroutine init_val_r1(r1, val)
      implicit none
      type(r1_t), pointer :: r1
      double precision, intent(in) :: val

      if (.not. associated(r1)) STOP "cannot initialize non existing r1"
      r1%vali(:)=val

      end subroutine init_val_r1
      !# </SUBROUTINE>
      !=================================================================

!DEBUG      !=================================================================
!DEBUG       !# <SUBROUTINE NAME=moy_r1>
!DEBUG       !# Ax=(Bx+Cx)/2
!DEBUG       subroutine moy_r1(Ax,Bx,Cx)
!DEBUG       implicit none
!DEBUG       type(r1_t), pointer :: Ax, Bx, Cx
!DEBUG       integer:: i
!DEBUG
!DEBUG       if (.not. associated(Ax)) STOP "moy: cannot use non existing Ax"
!DEBUG       if (.not. associated(Bx)) STOP "moy: cannot use non existing Bx"
!DEBUG       if (.not. associated(Cx)) STOP "moy: cannot use non existing Cx"
!DEBUG
!DEBUG       if ((Ax%dim1 .eq. Bx%dim1) .and. (Bx%dim1 .eq. Cx%dim1)) then
!DEBUG !$omp target teams distribute parallel do
!DEBUG          do i=1, Ax%dim1
!DEBUG             Ax%val(i)=0.5*(Bx%val(i)+Cx%val(i))
!DEBUG             Ax%val(i)=Ax%val(i)/3.14
!DEBUG          enddo
!DEBUG !$omp end target teams distribute parallel do
!DEBUG       else
!DEBUG         STOP "moy: Ax, Bx, Cx have different size"
!DEBUG       endif
!DEBUG
!DEBUG       end subroutine moy_r1
!DEBUG       !# </SUBROUTINE>
!DEBUG       !=================================================================

end module calcule_m

program maintest
   use array_m
   use calcule_m
   implicit none

   integer, parameter:: dim1=1000000
   type(r1_t), pointer :: r1list
   type(r1_t), pointer :: A, B, C
   integer :: i,fin
   logical :: jump=.true.

   call add_new_r1(r1list,dim1,"Tableau A",A)
   call add_new_r1(r1list,dim1,"Tableau B",B)
   call add_new_r1(r1list,dim1,"Tableau C",C)

   call init_val_r1(B,3.3_8)
   call init_i_r1(C)
   print*,'BEFORE'
   CALL print_r1(r1list)
   print*,'AFTER'
!                       DEBUG $omp target data map (to:B%val,C%val) map(from:A%val)
!   call moy_r1(A,B,C)
   fin= A%dim1
   !associate(PA=>A%val, PB=>B%val, PC=>C%val)
!$omp target data map(to:fin,B%vali,C%vali) map(from:A%vali)
!$omp target teams  distribute parallel do
         do i=1, fin
            A%vali(i)=0.5*(B%vali(i)+C%vali(i))
!            PA(i)=0.5*(PB(i)+PC(i))
         enddo
!$omp end target teams distribute parallel do
!$omp end target data
!   end associate

   CALL print_r1(r1list)

   ! checking
!   write(6,'(a,$)'),'OK '
!   do i=1,A%dim1
!      if (A%val(i).eq. 0.5*(B%val(i)+C%val(i))) then
!         if (.not. jump) then
!            jump=.true.
!            write(6,*)''
!            write(6,'(a,$)') 'OK '
!         endif
!         write(6,'(i0," ",$)') i
!      else
!         if (jump) then
!            jump=.false.
!            write(6,*)''
!            write(6,'(a,$)') 'BAD '
!         endif
!         write(6,'(i0," ",$)') i
!      endif
!   enddo
end program maintest
