module Device_C

  use iso_c_binding
  
  implicit none
  private
  
  public :: &
    AllocateTargetDouble, &
    AssociateTargetDouble, &
    DeallocateTarget, &
    DisassociateTarget
  
  interface 
    
    type ( c_ptr ) function AllocateTargetDouble ( nValues ) &
                              bind ( c, name = 'AllocateTargetDouble_OMP' )
      use iso_c_binding
      implicit none
      integer ( c_int ), value :: &
        nValues
    end function AllocateTargetDouble
    
     
    integer ( c_int ) function AssociateTargetDouble &
                        ( Host, Device, nValues, oValue ) &
                        bind ( c, name = 'AssociateTargetDouble_OMP' )
    
      use iso_c_binding
      implicit none
      type ( c_ptr ), value :: &
        Host, &
        Device
      integer ( c_int ), value :: &
        nValues, &
        oValue
    end function AssociateTargetDouble
    
    
    subroutine DeallocateTarget ( Device ) bind ( c, name = 'FreeTarget_OMP' )
      use iso_c_binding
      implicit none
      type ( c_ptr ), value :: &
        Device
    
    end subroutine DeallocateTarget

    
    integer ( c_int ) function DisassociateTarget ( Host ) &
                        bind ( c, name = 'DisassociateTarget_OMP' )
    
      use iso_c_binding
      implicit none
      type ( c_ptr ), value :: &
        Host
    end function DisassociateTarget
    
    
  end interface 

end module Device_C
