program main
   use iso_c_binding
   use Device_C
interface
subroutine testKernel ( A, B )
  real ( 8 ), dimension ( : ), intent ( inout ) :: &
    A
  real ( 8 ), dimension ( : ), intent ( in ) :: &
    B
 
end subroutine testKernel
end interface

  real ( 8 ), dimension ( :, : ), pointer, contiguous :: &
     dataArray => null (  )
  real ( 8 ), dimension ( :, : ), pointer :: &
      D_Scratch
  type (c_ptr)  :: D_dataArray = c_null_ptr  !-- Device pointer to dataArray
  integer :: i
  integer ( c_int ) :: Error
  allocate(dataArray(10,2))
  ! explicitly allocate memory on the device
  ! call OMP C runtime function:  omp_target_alloc
  D_dataArray = AllocateTargetDouble(20)
  call c_f_pointer (  D_dataArray, D_Scratch, [ 10, 2 ] )

  ! Make link between host and device data subarrays
  ! call OMP C runtime function: omp_target_associate_ptr
  do iV = 1, 2
     Error = AssociateTargetDouble ( c_loc(dataArray(:,iV)),  c_loc ( D_Scratch ( :, iV ) ), 10, 0 )
  end do

  do i = 1, size ( dataArray(:,1) )
     dataArray(i,1) = 3
  end do

  call testKernel(dataArray(:,2),dataArray(:,1))

  ! Check if 2nd column is not initialized by testKernel
  do i = 1, size ( dataArray(:,1) )
     if (dataArray(i,2) .eq. 3 ) then
        print *, "2nd column should not be initialized"
        stop 1
     end if
  end do

  associate &
      ( X  => dataArray ( :, 1 ) )
  do i = 1, size ( dataArray(:,1) )
     X(i)  = 5
  end do
  call testKernel(dataArray(:,2),dataArray(:,1))
  end associate
  ! Check if 2nd column is not initialized by testKernel
  do i = 1, size ( dataArray(:,1) )
     if (dataArray(i,2) .eq. 5 ) then
        print *, "2nd column should not be initialized"
        stop 1
     end if
  end do

  ! Clear link between host and device data subarrays
  ! call OMP C runtime function: omp_target_disassociate_ptr
  do iV = 1, 2
     Error = DisassociateTarget ( c_loc(dataArray(:,iV)))
  end do
  call DeallocateTarget(D_dataArray)
  deallocate(dataArray)
end program

subroutine testKernel ( A, B ) 
  real ( 8 ), dimension ( : ), intent ( inout ) :: &
    A
  real ( 8 ), dimension ( : ), intent ( in ) :: &
    B
  integer :: i
!$omp target parallel do 
  do i = 1, size ( A )
     A(i) = B(i)
  end do
 end subroutine testKernel

