subroutine tester (field, left_rcv_buffer, scalar)
  implicit none

  REAL(KIND=8),pointer :: field(:,:)
  REAL(KIND=8),pointer :: left_rcv_buffer(:)
  double precision :: scalar

  integer k,j,index
  integer y_min
  integer y_inc
  integer y_max
  integer depth
  integer buffer_offset
  integer x_min

  y_min = 1
  y_inc = 0
  y_max = 15360
  depth = 2
  buffer_offset = 0
  x_min = 1

!$OMP TARGET DATA MAP(to:left_rcv_buffer, field)
!$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO COLLAPSE(2)
    DO k=y_min-depth,y_max+y_inc+depth
      DO j=1,depth
        index= buffer_offset + j+(k+depth-1)*depth
        ! print *,"k,j,index,x_min-j = ",k,j,index,x_min-j
        field(x_min-j,k)=left_rcv_buffer(index)
      ENDDO
    ENDDO
!$OMP END TARGET DATA

!$OMP TARGET UPDATE FROM(field)
!$OMP TARGET UPDATE FROM(left_rcv_buffer)

  print *,"scalar=",scalar
  print *,"Array field=",field(-1,-1)
  print *,"Array left_rcv_buffer=",left_rcv_buffer(-1)

end subroutine tester


program main
  implicit none

  double precision :: scalar = 2

  REAL(KIND=8) ,pointer :: field(:,:)
  REAL(KIND=8) ,pointer :: left_rcv_buffer(:)

  interface
    subroutine tester(field, left_rcv_buffer, scalar)
      REAL(KIND=8) ,pointer :: field(:,:)
      REAL(KIND=8) ,pointer :: left_rcv_buffer(:)
      double precision scalar
    end subroutine
  end interface

  integer i,j
  allocate (field(-1:7682,-1:15362))
  allocate (left_rcv_buffer(307300))

!!$omp target enter data map(alloc: field)
!!$omp target enter data map(alloc: left_rcv_buffer)

  do j=-1,15362
    do i=-1,7682
      field(i,j) = 1.0
    enddo
  enddo

  do i=1,307300
    left_rcv_buffer(i) = 2.0
  enddo

  print *,"ptr to field = ", loc(field)
  print *,"ptr to left_rcv_buffer = ", loc(left_rcv_buffer)
  call tester(field, left_rcv_buffer, scalar)

  deallocate(field)
  deallocate(left_rcv_buffer)

end program
