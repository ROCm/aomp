module ttds_mod
  implicit none
  complex(8), pointer :: work2_c(:)
contains
  subroutine ttds(src)
    complex(8), intent(in), contiguous :: src(:,:,:)
    integer :: i, s1
    s1 = size(src,1)
    !$omp target teams distribute simd
    do i=1,s1
      work2_c(i)=src(i,1,1)
    end do
    !$omp end target teams distribute simd
  end subroutine
end module
