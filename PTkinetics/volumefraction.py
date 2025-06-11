#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:57:46 2023
last modified: Nov. 13, 2024
@author: Daniel N. Blaschke

This submodule provides functions to calculate the volume fraction of the second phase
due to nucleation on dislocations, grains, and homogeneous nucleation.
"""
from math import erfc # numba.jit supports this implementation but not the scipy one
import numpy as np
from scipy.integrate import quad, dblquad
# from scipy.special import erfc, exp1
from .data import kB, amu
from .utilities import jit, exp1

def relaxtime(VolumeFractions,times,min_VF=0.05,max_VF=0.95):
    '''takes arrays of voume fractions and times and determines the time between reaching min_VF and max_VF.'''
    if len(times[VolumeFractions<=max_VF])==0:
        return np.inf
    mask = (VolumeFractions>=min_VF) & (VolumeFractions<=max_VF)
    out = times[mask]
    if len(out)>0:
        out = out[-1]-out[0]
    else:
        out = 0.
    return out

@jit(nopython=True)
def fdis(alpha,epsilon=1e-2):
    '''ratio epsilon_dis/epsilon_hom (Dean's approximation)'''
    out = 0
    if alpha<1:
        out = (1-alpha)*(1-4*alpha/5)
    return max(out,epsilon)

def P_of_t_ramp(t,Pdot,Ptransition):
    '''pressure during ramp loading as a function of time, pressure rate, and transition pressure'''
    return Ptransition + Pdot*t

def t_of_P_ramp(P,Pdot,Ptransition):
    '''time during ramp loading as a function of pressure, pressure rate, and transition pressure'''
    return (P-Ptransition)/Pdot

def figrain(k,include_gb,include_ge,include_gc):
    '''determine coefficients f2, f1, and f0 needed to calculate nucleation on grain boundaries,
       edges, and corners'''
    if k<1 and include_gb:
        f2grain = 0.5*(2-3*k+k**2) #(1-k)**(5/3)
    else:
        f2grain = 0
    if k<np.sqrt(3)/2 and include_ge:
        f1grain = (1-2*k/np.sqrt(3))**2 ## using the approximation here
    else:
        f1grain = 0
    if k<np.sqrt(2/3) and include_gc:
        f0grain = (1-k/np.sqrt(2/3))**(5/2) ## using the approximation here
    else:
        f0grain = 0
    return (f2grain,f1grain,f0grain)

def sigrain(sym):
    '''coefficients s2, s1, s0 are currently only implemented for truncated octahedra (applicable for bcc crystals,
       bodycentered tetragonal crystals with c/a<sqrt(2), and some other cases)'''
    if sym=='trunc_oct':
        s2 = 0.75*(1+2*np.sqrt(3))
        s1 = 6*np.sqrt(2)
        s0 = 12
    else:
        raise ValueError("not implemented")
    return (s2,s1,s0)

def compute_prefactors(DeltaGPprime,rhomean,gammaAM,rhodis,burgers,shear,poisson,atommass,DeltaP=10,kappa=1e3,beta=1e-10):
    '''computes various quantities required to compute the volume fraction (fcts lambdaE_hd/grain()); required input parameters:
       DeltaGPprime [dimensionless] ... pressure derivative of the change in Gibbs free energy per volume between the two phases
       rhomean [kg/m^3] ... density at the transition pressure averaged between the two phases
       gammaAM [mJ/m^2] ... interfacial energy between the two phases
       rhodis [1/m^2] ... dislocation density
       burgers [m] ... Burgers vector length
       shear [GPa] .. shear modulus
       poisson [dimensionless] ...  Poisson's ratio
       DeltaP [GPa]
       kappa [m^3 / Js]
       beta [J/m]'''
    cpref = 2*kappa*np.sqrt(3*beta*DeltaGPprime*1e9*DeltaP)/DeltaP
    # rpref = cpref/2e4
    epshom = 16*np.pi/3*gammaAM**3/DeltaGPprime**2/1e27 # J GPa^2, only the prefactor of 1/(P-Ptransition)**2
    Ndotpref = 1e13*rhomean/(atommass*amu)/1e6/1e6 ## 1/1e6 for each s->mus and m^3->cm^3
    rhob2 = rhodis*burgers**2 # dimensionless
    mub2kap = 1e9*shear*burgers**2*(1-poisson/2)/(1-poisson) ## prefactor we need below in Pa m^2
    alpha_dis = DeltaGPprime*1e9*mub2kap/(2*np.pi**2*(gammaAM*1e-3)**2) # times (P-Ptransition)=Pdot*t
    return cpref,epshom,Ndotpref,rhob2,alpha_dis

@jit(nopython=True)
def integrand_hom(tp,t,Pdot,cpref,Ndotpref,epshom,cmax=np.inf,Ttarget=300):
    '''subroutine of lambda_hd'''
    if cmax==np.inf:
        rpref = cpref/2e4
    else:
        ## with limiting interface speed (will need ct as input):
        rpref = cmax*(1-np.exp(-cpref*(Pdot*tp)/cmax))/(Pdot*tp)/2e4
    a = np.log(rpref**3*Ndotpref*4*np.pi/3)
    # b = epshom/0.02585
    b = epshom/(kB*Ttarget)
    return np.exp(a-b/(Pdot*tp)**2)*(t**2-tp**2)**3

@jit(nopython=True)
def integrand_dis(tp,t,Pdot,cpref,Ndotpref,epshom,rhob2,alpha_dis,cmax=np.inf,Ttarget=300):
    '''subroutine of lambda_hd'''
    if cmax==np.inf:
        rpref = cpref/2e4
    else:
        ## with limiting interface speed (will need ct as input):
        rpref = cmax*(1-np.exp(-cpref*(Pdot*tp)/cmax))/(Pdot*tp)/2e4
    a = np.log((rpref)**3*Ndotpref*rhob2*4*np.pi/3)
    b = epshom*fdis(alpha_dis*Pdot*tp)/(kB*Ttarget)
    # if alpha_dis*Pdot*tp>1 and Pdot>1e-2:
    #     print(alpha_dis,Pdot,tp,alpha_dis*Pdot*tp)
    return np.exp(a-b/(Pdot*tp)**2)*(t**2-tp**2)**3

### integrate.quad options (to trade-off accuracy for speed in the numerical integrals)
quadepsabs=1.49e-04 ## absolute error tolerance; default: 1.49e-08
quadepsrel=1.49e-04 ## relative error tolerance; default: 1.49e-08
quadlimit=30 ## max no of subintervals; default: 50
def lambdaE_hd(t,Pdot,cpref,Ndotpref,epshom,rhob2,alpha_dis,cmax=np.inf,Ttarget=300,include_hom=True, include_disloc=True):
    '''volume fraction from hom. nucleation and nucl. on dislocations'''
    def lmbdE(t):
        if t>0 and (include_hom or include_disloc):
            if include_disloc and include_hom:
                out = quad(lambda tp: integrand_hom(tp,t,Pdot,cpref,Ndotpref,epshom,cmax=cmax,Ttarget=Ttarget)+integrand_dis(tp,t,Pdot,cpref,Ndotpref,epshom,rhob2,alpha_dis,cmax=cmax,Ttarget=Ttarget), 0, t, epsabs=quadepsabs, epsrel=quadepsrel, limit=quadlimit)
            elif include_hom:
                out = quad(lambda tp: integrand_hom(tp,t,Pdot,cpref,Ndotpref,epshom,cmax=cmax,Ttarget=Ttarget), 0, t, epsabs=quadepsabs, epsrel=quadepsrel, limit=quadlimit)
            elif include_disloc:
                out = quad(lambda tp: integrand_dis(tp,t,Pdot,cpref,Ndotpref,epshom,rhob2,alpha_dis,cmax=cmax,Ttarget=Ttarget), 0, t, epsabs=quadepsabs, epsrel=quadepsrel, limit=quadlimit)
        else:
            out = (0.,0.)
        return out[0]*Pdot**3
    if isinstance(t,float):
        out = lmbdE(t)
    else:
        out = np.asarray([lmbdE(t[ti]) for ti in range(len(t))])
    return out

@jit(nopython=True)
def integrandgrain2(x,t,epshom,f2grain,Ttarget,Pdot,rt0,Ndotpref,delta):
    '''subroutine of lambdaE_grain()'''
    A = epshom*f2grain/(kB*Ttarget)/Pdot**2
    ## integratedpiece = int_0^(t(1-x)) dtp ((1-tp/t)**2-x**2)*np.exp(-A/tp**2) which according to Mathematica is:
    tx2part = t**2*(x-1)**2
    integratedpiece_term1 = (np.exp(-A/(tx2part))/(3*t))*(x-1)*(2*A+(x-1)*(2*x+1)*t**2)
    integratedpiece_term2 = np.sqrt(A*np.pi)*(2*A+3*(x**2-1)*t**2)*erfc(np.sqrt(A)/(t*(1-x)))/(3*t**2)
    integratedpiece_term3 = (A/t)*exp1(A/tx2part)
    integratedpiece = integratedpiece_term1 + integratedpiece_term2 + integratedpiece_term3
    return np.exp(-np.pi*rt0**2*Ndotpref*delta*integratedpiece)

@jit(nopython=True)
def tpmaxgrain21(x,t):
    '''subroutine of lambdaE_grain()'''
    return t*(1-x)

@jit(nopython=True)
def integrandgrain1(tp,x,t,rt0,Ndotpref,delta,epshom,f1grain,Ttarget,Pdot):
    '''subroutine of lambdaE_grain()'''
    return x*np.exp(-2*rt0*Ndotpref*delta**2*np.sqrt((1-tp/t)**2-x**2)*np.exp(-epshom*f1grain/(kB*Ttarget)/(Pdot*tp)**2))

@jit(nopython=True)
def integrandgrain0(tp,t,epshom,f0grain,Ttarget,Pdot,Ndotpref,delta):
    '''subroutine of lambdaE_grain()'''
    A = epshom*f0grain/(kB*Ttarget)/(Pdot)**2
    integratedpiece = t*np.exp(-A/t**2) - np.sqrt(A*np.pi)*erfc(np.sqrt(A)/t)#  =int_0^t dt np.exp(-A/tp**2))
    return (1-tp/t)**3*Ndotpref*delta**3*np.exp(-A/tp**2)*np.exp(-Ndotpref*delta**3*integratedpiece)

def lambdaE_grain(t,Pdot,delta,D,cpref,Ndotpref,epshom,f2g,f1g,f0g,s2,s1,s0,cmax=np.inf,Ttarget=300):
    '''volume fraction from nucleation on grain boundaries, edges, and corners'''
    if cmax==np.inf:
        rpref = cpref/2e4
    else:
        ## with limiting interface speed (will need ct as input):
        rpref = cmax*(1-np.exp(-cpref*(Pdot*t)/cmax))/(Pdot*t)/2e4
    rt0 = rpref*(Pdot*t**2/2) ## c is approx. linear in t, hence the integration from 0 to t is trivially t^2/2
    def lmbdE(t):
        out = 0
        if f2g>0:
            lmbdout2 = quad(lambda x: integrandgrain2(x,t,epshom,f2g,Ttarget,Pdot,rt0,Ndotpref,delta),0.,1., epsabs=quadepsabs, epsrel=quadepsrel, limit=quadlimit)
            out += (2*s2*rt0/D)*(1-lmbdout2[0])
        if f1g>0:
            lmbdout1 = dblquad(lambda tp,x: integrandgrain1(tp,x,t,rt0,Ndotpref,delta,epshom,f1g,Ttarget,Pdot),0.,1.,0.,lambda x: tpmaxgrain21(x,t), epsabs=quadepsabs, epsrel=quadepsrel)
            out += (np.pi*s1*rt0**2/D**2)*(1-2*lmbdout1[0])
        if f0g>0:
            lmbdout0 = quad(lambda tp: integrandgrain0(tp,t,epshom,f0g,Ttarget,Pdot,Ndotpref,delta),0.,t, epsabs=quadepsabs, epsrel=quadepsrel, limit=quadlimit)
            out += (4*np.pi/3)*s0*(rt0**3/D**3)*lmbdout0[0]
        return out
    if t<0:
        return 0.
    if isinstance(t,float):
        out = lmbdE(t)
    else:
        raise ValueError("arrays are not supported by this fct.")
    return out
