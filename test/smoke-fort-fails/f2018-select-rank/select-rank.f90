! Example from: https://www.nag.com/nagware/np/r71_doc/nag_f2018.html#AUTOTOC_4
Program select_rank_example
  Integer :: a = 123, b(1,2) = Reshape( [ 10,20 ], [ 1,2 ] ), c(1,3,1) = 777, d(1,1,1,1,1)
  Call show(a)
  Call show(b)
  Call show(c)
  Call show(d)
Contains
  Subroutine show(x)
    Integer x(..)
    Select Rank(x)
    Rank (0)
      Print 1,'scalar',x
    Rank (1)
      Print 1,'vector',x
    Rank (2)
      Print 1,'matrix',x
    Rank (3)
      Print 1,'3D array',x
    Rank Default
      Print *,'Rank',Rank(x),'not supported'
    End Select
  1 Format(1x,a,*(1x,i0,:))
  End Subroutine
End Program
