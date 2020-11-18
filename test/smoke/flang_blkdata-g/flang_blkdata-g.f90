 block data bd_printcommon
      implicit none
      integer depth
      common /printcommon/ depth
      data depth /1/
      end

program foobar 
   print *,1
end program
