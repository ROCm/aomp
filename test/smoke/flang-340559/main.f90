program main
  use matrix
  implicit none
  integer*4 :: cnt, row, col, ip, i, j
  type(hmatrix_t) :: hmat
  type(matrix_t), pointer :: matp

  cnt = 3
  hmat%cnt = cnt
  allocate(hmat%mat(cnt))
  !$omp parallel do private(row, col, ip, i, j, matp) shared(hmat, cnt) default(none)
  do ip = 1, cnt
    matp => hmat%mat(ip)
    row = ip*2; col = ip
    allocate(matp%val(col, row))
    matp%row = row; matp%col = col
    do i = 1, col
      do j = 1, row
        matp%val(i,j) = i*0.5 + j
      enddo
    enddo
  enddo
  !$omp end parallel do
  do ip = 1, cnt
    if (ip > 1) write(*,*) ' '
    write(*,'(a,i2)') 'Submatrix', ip
    do i = 1, hmat%mat(ip)%col
      do j = 1, hmat%mat(ip)%row
        write(*,'(f5.1,a,$)') hmat%mat(ip)%val(i,j), ' '
      enddo
      write(*,*) ' '
    enddo
  enddo
  deallocate(hmat%mat)
  print *, "PASS"
  return
end program

! CHECK: Submatrix 1
! CHECK: 1.5   2.5
! CHECK: Submatrix 2
! CHECK: 1.5   2.5   3.5   4.5
! CHECK: 2.0   3.0   4.0   5.0
! CHECK: Submatrix 3
! CHECK: 1.5   2.5   3.5   4.5   5.5   6.5
! CHECK: 2.0   3.0   4.0   5.0   6.0   7.0
! CHECK: 2.5   3.5   4.5   5.5   6.5   7.5
