program main
  use matrix
  implicit none
  integer*4 :: cnt
  type(hmatrix_t) :: hmat_cpu, hmat_gpu
  logical :: check

  cnt = 32
  hmat_cpu = construct_hmatrix(cnt)
  hmat_gpu = copy_hmatrix(hmat_cpu)

  call matrix_product_on_cpu(hmat_cpu)
  call matrix_product_on_gpu(hmat_gpu)

  check = is_hmatrix_equal(hmat_cpu, hmat_gpu)
  if (check) then
    write(*,*) 'Success!'
  else
    write(*,*) 'Failure!'
  endif

  call destruct_hmatrix(hmat_cpu)
  call destruct_hmatrix(hmat_gpu)
end program
