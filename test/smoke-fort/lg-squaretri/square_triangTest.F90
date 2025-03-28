program square_triangTest
      USE OMP_LIB
 
      integer          :: i, m, n, k, k1, ld, iter, j_m_1
      double precision, pointer,contiguous :: Msq(:,:)
      double precision, pointer,contiguous :: Mtr(:)
 
      ld = 2000
      allocate(Msq(ld,ld))
      allocate(Mtr(ld*(ld+1)/2))
 
      do j = 1, ld
        do i = 1, ld
           Msq(i,j) = i*1.0 + j*ld*2.0
        enddo
      enddo
 
      !$omp target enter data  map(to:Msq)
      !$omp target enter data map(alloc:Mtr)
 
 
    DO iter = 1,5
      ! copy into low triangular with diagonal , column major 
      !$OMP TARGET TEAMS DISTRIBUTE PRIVATE(j_m_1, k1) map(to:Msq) map(from:Mtr)
      do j = 1, ld
        j_m_1 = j-1
        k = (ld * j_m_1 - (((j - 2) * j_m_1 ) / 2))  + 1
        !$OMP PARALLEL DO PRIVATE(j_m_1, k1) FIRSTPRIVATE(k)
        do i = j, ld
          k1 =  k + (i - j)
          Mtr(k1) = Msq(i,j)
        enddo
      enddo
    ENDDO

      !$omp target exit data map(delete:Msq)
      !$omp target exit data map(delete:Mtr)
      deallocate(Msq,Mtr)
 
      end program square_triangTest
