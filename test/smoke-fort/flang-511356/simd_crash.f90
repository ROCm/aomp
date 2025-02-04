program test_simd
  INTEGER I
  INTEGER,PARAMETER :: K = 1024
  REAL A(K)

  I = 0
  IF (I==0) THEN
    A = 1.0
  ENDIF

  !$OMP SIMD
  DO I=1,K
    A(I)=A(I)+I
  ENDDO
end program test_simd
