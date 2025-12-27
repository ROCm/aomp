! The initial version of the test case was taken from Fujitsu test suite:
! https://github.com/fujitsu/compiler-test-suite/blob/main/Fortran/0614/0614_0005.f
! Fujitsu test suite is ARM-focused open source test suite:
! https://github.com/fujitsu/compiler-test-suite
! The original version does not explicitly state that the Cray pointer
! from sub2 can alias, leading to miscompilation on the ARM platform
! because Flang assumes that Cray pointers never alias.
! This version includes the target attribute for the variable 'a',
! which prevents miscompilation on ARM platforms.
! See https://github.com/llvm/llvm-project/issues/141928 for more details.

module mymodule
contains
      subroutine sub2(a,b)
      integer, target:: a(10)
      integer b(10),pa,pb
      pointer (p1 ,pa)
      pointer (p2 ,pb)

      p1 = loc(a(1))
      p2 = loc(b(1))

      do i = 1,9
         pa = pa + a(i+1)
         p1 = p1 + 4
      end do

      b(1:10) = a(1:10) + b(1:10)

      return
      end
end module mymodule

      program main
      use mymodule
      structure /str1/
        integer*4  ia(10)/10,9,8,7,6,5,4,3,2,1/
        integer*4  ib(10)/1,2,3,4,5,6,7,8,9,10/
      end structure
      structure /str2/
        integer*4  ia(10)
        integer*4  ib(10)
      end structure
      structure /str3/
        integer*4  ia(10)/ 3, 5, 7, 9,11,13,15,17,19,10/
        integer*4  ib(10)/13,14,15,16,17,18,19,20,21,11/
      end structure

      record /str1/ z
      record /str2/ zp
      record /str3/ x
      pointer (p,zp)
      integer,allocatable:: alc(:,:)
      integer za(10),zb(10),err/0/

      p = loc(z)
      call sub(zp.ia ,zp.ib)
      
      allocate(alc(z.ia(10),z.ib(9)))
      alc = 0
      alc(:,1) = z.ia
      alc(:,2) = z.ib
      za = alc(:,1)
      zb = alc(:,2)
      call sub2(za,zb)
      alc(:,1) = za
      alc(:,2) = zb
      do 10 i=1,10
         if ( x.ia(i) .ne. alc(i,1) )err=1
         if ( x.ib(i) .ne. alc(i,2) )err=1
 10   continue
      if (err .eq. 0) then
         write(6,*)'*** ok ***'
      else
         write(6,*)'*** ng ***'
         write(6,*)alc
         stop 1
      endif

      deallocate(alc)
      end

      subroutine sub(a,b)
      integer,allocatable:: x(:,:)
      integer*4 a(10),b(10)
      
      allocate(x(b(10),b(2)))
      
      x(:,1) = a
      x(:,2) = b
      x(:,1) = x(:,1) + x(:,2)
      x(:,2) = x(:,1) - x(:,2)
      x(:,1) = x(:,1) - x(:,2)
      a      = x(:,1)
      b      = x(:,2)

      deallocate (x)
      return
      end

