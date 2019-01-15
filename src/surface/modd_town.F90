!-------------------------------------- LICENCE BEGIN ------------------------------------
!Environment Canada - Atmospheric Science and Technology License/Disclaimer, 
!                     version 3; Last Modified: May 7, 2008.
!This is free but copyrighted software; you can use/redistribute/modify it under the terms 
!of the Environment Canada - Atmospheric Science and Technology License/Disclaimer 
!version 3 or (at your option) any later version that should be found at: 
!http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html 
!
!This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
!without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
!See the above mentioned License/Disclaimer for more details.
!You should have received a copy of the License/Disclaimer along with this software; 
!if not, you can write to: EC-RPN COMM Group, 2121 TransCanada, suite 500, Dorval (Quebec), 
!CANADA, H9P 1J3; or send e-mail to service.rpn@ec.gc.ca
!-------------------------------------- LICENCE END --------------------------------------
!     ################
      MODULE MODD_TOWN
!     ################
!
!!****  Initialisation des variables necessaires pour TEB      
!!
!-------------------------------------------------------------------------------
!
!*       0.   DECLARATIONS
!             ------------
!
implicit none 
!
common/ttte/XMASK,XINI_LAT,XINI_LON,XINI_ZS,XINI_TSOIL,XINI_TS, &
XINI_TA,XINI_QA,XTOWN,XQFVH,XQFRE,XQFRH,XQFCE,XQFCH,XQFIE,XQFIH,&
XQ_TOWN,XU_CANYON,XRN_ROOF,XH_ROOF,                             &
XLE_ROOF,XLES_ROOF,XGFLUX_ROOF,XRUNOFF_ROOF,XRN_ROAD,XH_ROAD,   &
XLE_ROAD,XLES_ROAD,XGFLUX_ROAD,XRUNOFF_ROAD,XRN_WALL,XH_WALL,   &
XLE_WALL,XGFLUX_WALL,XRNSNOW_ROOF,XHSNOW_ROOF,XLESNOW_ROOF,     &
XGSNOW_ROOF,XMELT_ROOF,XRNSNOW_ROAD,XHSNOW_ROAD,XLESNOW_ROAD,   &
XGSNOW_ROAD,XMELT_ROAD,XRN,XH,XLE,XGFLUX,XEVAP,XRUNOFF,XCH,XRI,XUSTAR
!$OMP THREADPRIVATE(/ttte/)
#define ALLOCATABLE POINTER
!
INTEGER                           :: NNI             ! Number of grid points
REAL, DIMENSION(:)  , ALLOCATABLE :: XMASK           ! Land/sea maks          
REAL, DIMENSION(:)  , ALLOCATABLE :: XINI_LAT        ! Latitude              
REAL, DIMENSION(:)  , ALLOCATABLE :: XINI_LON        ! Longitude             
REAL, DIMENSION(:)  , ALLOCATABLE :: XINI_ZS         ! Topography            
REAL, DIMENSION(:)  , ALLOCATABLE :: XINI_TSOIL      ! Soil temperature     
REAL, DIMENSION(:)  , ALLOCATABLE :: XINI_TS         ! Surface temperature   
REAL, DIMENSION(:)  , ALLOCATABLE :: XINI_TA         ! Air temperature at first level
REAL, DIMENSION(:)  , ALLOCATABLE :: XINI_QA         ! Air specific humidity at first level
REAL, DIMENSION(:)  , ALLOCATABLE :: XTOWN           ! Total fraction of urban cover
!
REAL, DIMENSION(:)  , ALLOCATABLE :: XQFVH           ! Energy concumption for traffic
REAL, DIMENSION(:)  , ALLOCATABLE :: XQFRE           ! Elec concumption for residential areas
REAL, DIMENSION(:)  , ALLOCATABLE :: XQFRH           ! Fuel concumption for residential areas
REAL, DIMENSION(:)  , ALLOCATABLE :: XQFCE           ! Elec concumption for commercial areas
REAL, DIMENSION(:)  , ALLOCATABLE :: XQFCH           ! Fuel concumption for commercial areas
REAL, DIMENSION(:)  , ALLOCATABLE :: XQFIE           ! Elec concumption for industrial areas
REAL, DIMENSION(:)  , ALLOCATABLE :: XQFIH           ! Fuel concumption for industrial areas
!
REAL, DIMENSION(:)  , ALLOCATABLE :: XQ_TOWN         ! Town averaged Specific humidity
REAL, DIMENSION(:)  , ALLOCATABLE :: XU_CANYON       ! Wind in canyon         
REAL, DIMENSION(:)  , ALLOCATABLE :: XRN_ROOF        ! Net radiation on roof 
REAL, DIMENSION(:)  , ALLOCATABLE :: XH_ROOF         ! Sensible heat flux on roof
REAL, DIMENSION(:)  , ALLOCATABLE :: XLE_ROOF        ! Latent heat flux on roof
REAL, DIMENSION(:)  , ALLOCATABLE :: XLES_ROOF       ! Sublimation heat flux on roof
REAL, DIMENSION(:)  , ALLOCATABLE :: XGFLUX_ROOF     ! Storage heat flux on roof
REAL, DIMENSION(:)  , ALLOCATABLE :: XRUNOFF_ROOF    ! Water runoff from roof        
REAL, DIMENSION(:)  , ALLOCATABLE :: XRN_ROAD        ! Net radiation on road                
REAL, DIMENSION(:)  , ALLOCATABLE :: XH_ROAD         ! Sensible heat flux on road     
REAL, DIMENSION(:)  , ALLOCATABLE :: XLE_ROAD        ! Latent heat flux on road            
REAL, DIMENSION(:)  , ALLOCATABLE :: XLES_ROAD       ! Sublimation heat flux on road
REAL, DIMENSION(:)  , ALLOCATABLE :: XGFLUX_ROAD     ! Storage heat flux on road       
REAL, DIMENSION(:)  , ALLOCATABLE :: XRUNOFF_ROAD    ! Water runoff from road 
REAL, DIMENSION(:)  , ALLOCATABLE :: XRN_WALL        ! Net radiation on wall                
REAL, DIMENSION(:)  , ALLOCATABLE :: XH_WALL         ! Sensible heat flux on wall     
REAL, DIMENSION(:)  , ALLOCATABLE :: XLE_WALL        ! Latent heat flux on wall            
REAL, DIMENSION(:)  , ALLOCATABLE :: XGFLUX_WALL     ! Storage heat flux on wall            
REAL, DIMENSION(:)  , ALLOCATABLE :: XRNSNOW_ROOF    ! Net radiation over snow            
REAL, DIMENSION(:)  , ALLOCATABLE :: XHSNOW_ROOF     ! Sensible heat flux over snow  
REAL, DIMENSION(:)  , ALLOCATABLE :: XLESNOW_ROOF    ! Latent heat flux over snow   
REAL, DIMENSION(:)  , ALLOCATABLE :: XGSNOW_ROOF     ! Flux under snow             
REAL, DIMENSION(:)  , ALLOCATABLE :: XMELT_ROOF      ! Snow melt             
REAL, DIMENSION(:)  , ALLOCATABLE :: XRNSNOW_ROAD    ! Net radiation over snow            
REAL, DIMENSION(:)  , ALLOCATABLE :: XHSNOW_ROAD     ! Sensible heat flux over snow  
REAL, DIMENSION(:)  , ALLOCATABLE :: XLESNOW_ROAD    ! Latent heat flux over snow     
REAL, DIMENSION(:)  , ALLOCATABLE :: XGSNOW_ROAD     ! Flux under snow        
REAL, DIMENSION(:)  , ALLOCATABLE :: XMELT_ROAD      ! Snow melt 
REAL, DIMENSION(:)  , ALLOCATABLE :: XRN             ! Net radiation over town 
REAL, DIMENSION(:)  , ALLOCATABLE :: XH              ! Sensible heat flux over town
REAL, DIMENSION(:)  , ALLOCATABLE :: XLE             ! Latent heat flux over town     
REAL, DIMENSION(:)  , ALLOCATABLE :: XGFLUX          ! Storage heat flux over town          
REAL, DIMENSION(:)  , ALLOCATABLE :: XEVAP           ! Evaporation                    
REAL, DIMENSION(:)  , ALLOCATABLE :: XRUNOFF         ! Runoff over ground   
REAL, DIMENSION(:)  , ALLOCATABLE :: XCH             ! Heat drag             
REAL, DIMENSION(:)  , ALLOCATABLE :: XRI             ! Richardson number               
REAL, DIMENSION(:)  , ALLOCATABLE :: XUSTAR          ! Friction velocity              

!
END MODULE MODD_TOWN
