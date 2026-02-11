#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:57:46 2023
last modified: Feb. 5, 2026
@author: Daniel N. Blaschke

This submodule provides functions to calculate the volume fraction of the second phase
due to nucleation on dislocations, grains, and homogeneous nucleation.
"""
import sys
import os
import numpy as np
from scipy.integrate import quad, dblquad

def compilefortranmodule():
    '''compiles the Fortran subroutines if a Fortran compiler is available'''
    cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    exitcode = os.system('python -m numpy.f2py -c fraction_subroutines.f90 -m fraction_subroutines')
    os.chdir(cwd)
    return exitcode

try:
    from .fraction_subroutines import compute_prefactors, integrand_hom, integrand_dis, integrandgrain2, tpmaxgrain21, integrandgrain1, integrandgrain0
    from .fraction_subroutines import f2grain, f1grain, f0grain
except ImportError:
    print("ERROR importing Fortran sub-module 'fraction_subroutines'; attempting to recompile ...\n")
    if compilefortranmodule()==0:
        print("\nSUCCESS - please rerun/reload this script")
        sys.exit()
    else:
        print("\nFAILED to compile Fortran sub-module 'fraction_subroutines' - please check the f2py logs")

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

def P_of_t_ramp(t,Pdot,Ptransition):
    '''pressure during ramp loading as a function of time, pressure rate, and transition pressure'''
    return Ptransition + Pdot*t

def t_of_P_ramp(P,Pdot,Ptransition):
    '''time during ramp loading as a function of pressure, pressure rate, and transition pressure'''
    return (P-Ptransition)/Pdot

def figrain(k,include_gb,include_ge,include_gc):
    '''determine coefficients f2, f1, and f0 needed to calculate nucleation on grain boundaries,
       edges, and corners'''
    f2g = 0
    if include_gb:
        f2g = f2grain(k)
    f1g = 0
    if include_ge:
        f1g = f1grain(k)
    f0g = 0
    if include_gc:
        f0g = f0grain(k)
    return (f2g,f1g,f0g)

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

### integrate.quad options (to trade-off accuracy for speed in the numerical integrals)
quadepsabs=1.49e-04 ## absolute error tolerance; default: 1.49e-08
quadepsrel=1.49e-04 ## relative error tolerance; default: 1.49e-08
quadlimit=30 ## max no of subintervals; default: 50
def lambdaE_hd(t,Pdot,rpref,Ndotpref,epshom,rhob2,alpha_dis,Ttarget=300,include_hom=True, include_disloc=True):
    '''volume fraction from hom. nucleation and nucl. on dislocations'''
    def lmbdE(t):
        out = (0.,0.)
        if t>0 and (include_hom or include_disloc):
            if include_disloc and include_hom:
                out = quad(lambda tp: integrand_hom(tp,t,Pdot,rpref,Ndotpref,epshom,Ttarget)
                           +integrand_dis(tp,t,Pdot,rpref,Ndotpref,epshom,rhob2,alpha_dis,Ttarget), 0, t, epsabs=quadepsabs, epsrel=quadepsrel, limit=quadlimit)
            elif include_hom:
                out = quad(lambda tp: integrand_hom(tp,t,Pdot,rpref,Ndotpref,epshom,Ttarget), 0, t, epsabs=quadepsabs, epsrel=quadepsrel, limit=quadlimit)
            elif include_disloc:
                out = quad(lambda tp: integrand_dis(tp,t,Pdot,rpref,Ndotpref,epshom,rhob2,alpha_dis,Ttarget), 0, t, epsabs=quadepsabs, epsrel=quadepsrel, limit=quadlimit)
        return out[0]*Pdot**3
    if isinstance(t,float):
        out = lmbdE(t)
    else:
        out = np.asarray([lmbdE(t[ti]) for ti in range(len(t))])
    return out

def lambdaE_grain(t,Pdot,delta,D,rpref,Ndotpref,epshom,f2g,f1g,f0g,s2,s1,s0,Ttarget=300):
    '''volume fraction from nucleation on grain boundaries, edges, and corners'''
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
