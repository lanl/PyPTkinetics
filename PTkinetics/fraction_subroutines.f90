! Author: Daniel N. Blaschke
! Date: Nov. 13, 2024
! requires a Fortran 2008 or later (using complementary error function)

module parameters
implicit none
integer,parameter :: sel = selected_real_kind(10)
real(kind=sel), parameter :: kB = 1.38064852d-23        ! Boltzmann constant
real(kind=sel), parameter :: pi = (4.d0*atan(1.d0)) ! pi
real(kind=sel), parameter :: pi2 = (4.d0*atan(1.d0))**2 ! pi squared
real(kind=sel), parameter :: amu = 1.66053904d-27 ! atomic mass unit [kg]
real(kind=sel), parameter :: avogadro = 6.02214076d23 ! 1/mol
end module parameters

subroutine version(versionnumber)
  integer, intent(out) :: versionnumber
  versionnumber=20241113
end subroutine version

SUBROUTINE exp1(x,approxE1)
! this implementation of the exponential integral E1(x) for real numbers uses approximations
! 5.1.53 (0<=x<1) and 5.1.56 (1<=x<inf) from the Handbook of Mathematical Functions
! by Abramovitz and Stegun (9th printing)
!-----------------------------------------------------------------------
  use, intrinsic :: ieee_arithmetic
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: x
  REAL(KIND=sel), INTENT(OUT) :: approxE1
  REAL(KIND=sel) :: numer, denom
  
  if (0.d0<=x .and. x<1) then
    approxE1 = - log(x) - 0.57721566d0 + 0.99999193d0*x - 0.24991055d0*x**2 &
               + 0.05519968d0*x**3 - 0.00976004d0*x**4 + 0.00107857d0*x**5
  else if (1<=x) then
    numer = (x**4+8.5733287401d0*x**3+18.0590169730d0*x**2+8.6347608925d0*x+0.2677737343d0)
    denom = (x**4+9.5733223454d0*x**3+25.6329561486d0*x**2+21.0996530827d0*x+3.9584969228d0)
    approxE1 = (numer/denom)*exp(-x)/x
  else
    approxE1 = ieee_value( x, ieee_signaling_nan )
  endif
  
END SUBROUTINE exp1
    
SUBROUTINE fdis(alpha,eps,fdisloc)
! determine a model parameter for nucleation on dislocations
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: alpha,eps
  REAL(KIND=sel), INTENT(OUT) :: fdisloc
  
  fdisloc = eps
  if (alpha<1.d0) then
    fdisloc = max((1-alpha)*(1-4*alpha/5),eps)
  endif
  
  RETURN
END SUBROUTINE fdis

SUBROUTINE f2grain(k,f2)
! determine a model parameter for nucleation on grain boundaries
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: k
  REAL(KIND=sel), INTENT(OUT) :: f2
 
  f2=0.d0
  if (k<1.d0) then
    f2 = 0.5d0*(2.d0-3.d0*k+k**2)
  endif
  
  RETURN
END SUBROUTINE f2grain

SUBROUTINE f1grain(k,f1)
! determine a model parameter for nucleation on grain boundaries
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: k
  REAL(KIND=sel), INTENT(OUT) :: f1
 
  f1=0.d0
  if (k<sqrt(3.d0)/2.d0) then
    f1 = (1.d0-2.d0*k/sqrt(3.d0))**2
  endif
  
  RETURN
END SUBROUTINE f1grain

SUBROUTINE f0grain(k,f0)
! determine a model parameter for nucleation on grain boundaries
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: k
  REAL(KIND=sel), INTENT(OUT) :: f0
 
  f0=0.d0
  if (k<sqrt(2.d0/3.d0)) then
    f0 = (1-k/sqrt(2.d0/3.d0))**(5.d0/2.d0)
  endif
  
  RETURN
END SUBROUTINE f0grain

SUBROUTINE rpref(kappa,beta,DeltaGPprime,DeltaP,rprefactor)
! determine a model parameter for nucleation
! units:
!   kappa [m^3 / Js]
!   beta [J/m]
!   DeltaGPprime [dimensionless]
!   DeltaP [GPa]
!   rprefactor [cm / GPa musec
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: kappa, beta, DeltaGPprime, DeltaP
  REAL(KIND=sel), INTENT(OUT) :: rprefactor
 
  rprefactor = 2.d0*kappa*sqrt(3.d0*beta*DeltaGPprime*1.d9*DeltaP)/DeltaP/2.d4
  
  RETURN
END SUBROUTINE rpref


SUBROUTINE epshom(gammaAM,DeltaGPprime,epsilonhom)
! determine a model parameter for nucleation
! units:
!   gammaAM [mJ/m^2]
!   DeltaGPprime [dimensionless]
!   epshom [J GPa^2]
!-----------------------------------------------------------------------
  use parameters, only : pi
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: gammaAM, DeltaGPprime
  REAL(KIND=sel), INTENT(OUT) :: epsilonhom
 
  epsilonhom = 16.d0*pi/3*gammaAM**3.d0/DeltaGPprime**2/1d27 ! J GPa^2, only the prefactor of 1/(P-Ptransition)**2
  
  RETURN
END SUBROUTINE epshom

SUBROUTINE Ndotpref(rhomean,atommass,Ndotprefactor)
! determine a model parameter for nucleation
! units:
!   rhomean [kg/m^3] ... mean density
!   atommass [atomic mass units]
!   Ndotpref []
!-----------------------------------------------------------------------
  use parameters, only : amu
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: rhomean,atommass
  REAL(KIND=sel), INTENT(OUT) :: Ndotprefactor
  ! assume here Debye frequency ~ 1d13 /s,
  ! then 1/1d6 for each s->mus and m^3->cm^3 yields prefactor 1d13/1d12=10
  Ndotprefactor = 10.d0*rhomean/(atommass*amu)
  
  RETURN
END SUBROUTINE Ndotpref

SUBROUTINE compute_prefactors(DeltaGPprime,rhomean,gammaAM,rhodis,burgers,&
           shear,poisson,atommass,DeltaP,kappa,beta,rpref,epshom,Ndotpref,rhob2,alpha_dis)
! determine a model parameter for nucleation
! units:
!   DeltaGPprime [dimensionless]
!   rhomean [kg/m^3] ... mean density
!   gammaAM [mJ/m^2]
!   rhodis [1/m^2]
!   burgers [m]
!   shear [GPa]
!   poisson [dimensionless]
!   atommass [atomic mass units]
!   kappa [m^3 / Js]
!   beta [J/m]
!   DeltaGPprime [dimensionless]
!   DeltaP [GPa]
!
!   rpref [cm / GPa musec]
!   epshom [J GPa^2]
!   Ndotpref []
!   rhob2 [dimensionless]
!   alpha_dis
!-----------------------------------------------------------------------
  use parameters, only : pi, amu
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: DeltaGPprime,rhomean,gammaAM,rhodis,burgers
  REAL(KIND=sel), INTENT(IN)  :: shear,poisson,atommass,DeltaP,kappa,beta
  REAL(KIND=sel), INTENT(OUT) :: rpref,epshom,Ndotpref,rhob2,alpha_dis
  REAL(KIND=sel)  :: mub2kap
 
  rpref = 2.d0*kappa*sqrt(3.d0*beta*DeltaGPprime*1.d9*DeltaP)/DeltaP/2.d4
  epshom = 16.d0*pi/3*gammaAM**3.d0/DeltaGPprime**2/1d27 ! J GPa^2, only the prefactor of 1/(P-Ptransition)**2
  ! assume here Debye frequency ~ 1d13 /s,
  ! then 1/1d6 for each s->mus and m^3->cm^3 yields prefactor 1d13/1d12=10
  Ndotpref = 10.d0*rhomean/(atommass*amu)
  rhob2 = rhodis*burgers**2
  mub2kap = 1.d9*shear*burgers**2*(1.d0-0.5d0*poisson)/(1.d0-poisson) ! prefactor we need below in Pa m^2
  alpha_dis = DeltaGPprime*1.d9*mub2kap/(2.d0*pi**2*(gammaAM*1.d-3)**2) ! times (P-Ptransition)=Pdot*t
  
  RETURN
END SUBROUTINE compute_prefactors

SUBROUTINE integrand_hom(tp,t,Pdot,rpref,Ndotpref,epshom,Ttarget,integrand_homogen)
! subroutine of lambda_hd
!-----------------------------------------------------------------------
  use parameters, only : pi, kB
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: tp,t,Pdot,rpref,Ndotpref,epshom,Ttarget
  REAL(KIND=sel), INTENT(OUT) :: integrand_homogen
  REAL(KIND=sel)  :: a, b
  
  a = log(rpref**3*Ndotpref*4.d0*pi/3.d0)
  b = epshom/(kB*Ttarget)
  integrand_homogen = exp(a-b/(Pdot*tp)**2)*(t**2-tp**2)**3
  
END SUBROUTINE integrand_hom

SUBROUTINE integrand_dis(tp,t,Pdot,rpref,Ndotpref,epshom,rhob2,alpha_dis,Ttarget,&
                         integrand_disloc)
! subroutine of lambda_hd
!-----------------------------------------------------------------------
  use parameters, only : pi, kB
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: tp,t,Pdot,rpref,Ndotpref,epshom,rhob2,alpha_dis,Ttarget
  REAL(KIND=sel), INTENT(OUT) :: integrand_disloc
  REAL(KIND=sel)  :: a, b, fd
  
  a = log(rpref**3*Ndotpref*rhob2*4.d0*pi/3.d0)
  call fdis(alpha_dis*Pdot*tp,1.d-2,fd)
  b = epshom*fd/(kB*Ttarget)
  integrand_disloc = exp(a-b/(Pdot*tp)**2)*(t**2-tp**2)**3
  
END SUBROUTINE integrand_dis


SUBROUTINE integrandgrain2(x,t,epshom,f2grain,Ttarget,Pdot,rt0,Ndotpref,delta,&
                         integrand_g2)
! subroutine of lambda_grain
!-----------------------------------------------------------------------
  use parameters, only : pi, kB
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: x,t,epshom,f2grain,Ttarget,Pdot,rt0,Ndotpref,delta
  REAL(KIND=sel), INTENT(OUT) :: integrand_g2
  REAL(KIND=sel)  :: A, tx2part, term1, term2, term3, integratedpiece
  
  A = epshom*f2grain/(kB*Ttarget)/Pdot**2
  tx2part = t**2*(x-1)**2
  term1 = (exp(-A/(tx2part))/(3*t))*(x-1)*(2*A+(x-1)*(2*x+1)*t**2)
  term2 = sqrt(A*pi)*(2*A+3*(x**2-1)*t**2)*erfc(sqrt(A)/(t*(1-x)))/(3*t**2)
  call exp1(A/tx2part,term3)
  integratedpiece = term1 + term2 + (A/t)*term3
  integrand_g2 = exp(-pi*rt0**2*Ndotpref*delta*integratedpiece)
  
END SUBROUTINE integrandgrain2

SUBROUTINE tpmaxgrain21(x,t,tpmax)
! subroutine of lambda_grain
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
  REAL(KIND=sel), INTENT(IN)  :: x,t
  REAL(KIND=sel), INTENT(OUT)  :: tpmax
  tpmax = t*(1-x)
END SUBROUTINE tpmaxgrain21

SUBROUTINE integrandgrain1(tp,x,t,rt0,Ndotpref,delta,epshom,f1grain,Ttarget,Pdot,&
                         integrand_g1)
! subroutine of lambda_grain
  use parameters, only : kB
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: tp,x,t,rt0,Ndotpref,delta,epshom,f1grain,Ttarget,Pdot
  REAL(KIND=sel), INTENT(OUT) :: integrand_g1
  
  integrand_g1 = x*exp(-2*rt0*Ndotpref*delta**2*sqrt((1-tp/t)**2-x**2)*exp(-epshom*f1grain/(kB*Ttarget)/(Pdot*tp)**2))
END SUBROUTINE integrandgrain1

SUBROUTINE integrandgrain0(tp,t,epshom,f0grain,Ttarget,Pdot,Ndotpref,delta,&
                         integrand_g0)
! subroutine of lambda_grain
  use parameters, only : pi, kB
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: tp,t,epshom,f0grain,Ttarget,Pdot,Ndotpref,delta
  REAL(KIND=sel), INTENT(OUT) :: integrand_g0
  REAL(KIND=sel)  :: A, integratedpiece
  
  A = epshom*f0grain/(kB*Ttarget)/(Pdot)**2
  integratedpiece = t*exp(-A/t**2) - sqrt(A*pi)*erfc(sqrt(A)/t)
  integrand_g0 = (1-tp/t)**3*Ndotpref*delta**3*exp(-A/tp**2)*exp(-Ndotpref*delta**3*integratedpiece)
END SUBROUTINE integrandgrain0

