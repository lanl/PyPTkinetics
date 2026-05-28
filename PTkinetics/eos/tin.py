#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:57:46 2023
last modified: July 18, 2024
@author: Daniel N. Blaschke

Vinet EOS for beta and gamma tin using parametrization of Ann Wills (based on Carl Greeff's tabular EOS), see LA-UR-23-26917
"""

import numpy as np
# from numba import jit
from scipy import optimize

# all tuples contain values for beta phase and gamma phase (in this order):
phases = ('beta','gamma')
rho0 = (7285,7271) # density at Tref and zero pressure in kg/m^3
Tref = (298,298) # ref. temperature in K
B0 = (52.90e9,38.78e9) # bulk modulus in Pa
dBT_dP_0 = (5.3345,6.0532) # change in B with pressure at ref. state
## will need:
eta0 = np.round((3/2)*(np.array(dBT_dP_0)-1),6)
alpha0 = (7.2977e-5,10.85405e-5) # thermal expansion coeff. in units of 1/K
CV0 = (214.9,216.1) # specific heat in J/kg/K
E0 = (0.0658e6,0.1025e6) # internal energy at ref. state in J/kg
S0 = (441.9,505.1) # entropy  at ref. state in J/kg/K

def phasearray(arg):
    '''ensure arg is a 2d array and turn it into one otherwise'''
    if isinstance(arg,np.ndarray) and len(arg)==1:
        arg = np.array([arg[0],arg[0]])
    if isinstance(arg,float):
        arg = np.array([arg,arg])
    return arg

# @jit(nopython=True)
def Pressure(rho,T):
    '''returns pressure as a function of density and temperature for both phases'''
    out = np.zeros((2))
    rho = phasearray(rho)
    for i in range(2):
        X = np.cbrt(rho0[i]/rho[i])
        Pref = 3*B0[i]*(1-X)*np.exp(eta0[i]*(1-X))/X**2
        if isinstance(Pref,np.ndarray):
            Pref = Pref[0]
        out[i] = Pref + alpha0[i]*B0[i]*(T-Tref[i])
    return out

def Density(P,T):
    '''returns density as a function of pressure and temperature for both phases'''
    out = np.zeros((2))
    for i in range(2):
        def f(x):
            return abs(Pressure(x,T)[i]-P)
        rho = optimize.fsolve(f, rho0[i])
        out[i] = rho[0]
    return out

def Entropy(rho,T):
    '''returns entropy as a function of density and temperature for both phases'''
    rho = phasearray(rho)
    out = np.zeros(2)
    for i in range(2):
        out[i] = S0[i] + alpha0[i]*B0[i]*(1/rho[i]-1/rho0[i]) + CV0[i]*np.log(T/Tref[i])
    return out

def Uint(rho,T):
    '''returns internal energy as a function of density and temperature'''
    out = np.zeros(2)
    rho = phasearray(rho)
    for i in range(2):
        X = np.cbrt(rho0[i]/rho[i])
        Z = 1-X
        out[i] = E0[i] + 9*B0[i]*(1-np.exp(eta0[i]*Z)*(1-eta0[i]*Z))/(eta0[i]**2*rho0[i]) - \
            alpha0[i]*B0[i]*(1-rho0[i]/rho[i])*Tref[i]/rho0[i] + \
            CV0[i]*(T-Tref[i])
    return out

def Gibbs(P,T):
    '''returns the Gibbs free energy as a function of pressure and temperature for both phases'''
    out = np.zeros(2)
    rho = Density(P,T)
    E = Uint(rho,T)
    S = Entropy(rho,T)
    if isinstance(P,np.ndarray) and len(P)==1:
        P = P[0]
    for i in range(2):
        out[i] = E[i] - S[i]*T + P/rho[i]
    return out

def DeltaGibbs(P,T):
    '''returns the difference in Gibbs free energy of the two phases (gamma and beta tin) as a function of pressure and temperature'''
    G = Gibbs(P,T)
    return G[0]-G[1]

def Pcoex(T):
    '''determines the transition pressure between the two phases (gamma and beta tin) as a function of temperature'''
    def f(x):
        return abs(DeltaGibbs(x,T))
    return optimize.fsolve(f, 1e10)[0]

def DeltaGPprime(P,T,dP=1e3):
    '''calculates the pressure derivative of the difference in Gibbs free energy of the two phases as a function of pressure and temperature'''
    Gplus = DeltaGibbs(P+dP/2,T)
    Gmin = DeltaGibbs(P-dP/2,T)
    return (Gplus-Gmin)/dP # J/kg Pa

Ptrans300 = Pcoex(300)
rho_coex300 = Density(Ptrans300,300)
rhomean_coex300 = sum(rho_coex300)/2 ## average density directly
# rhomean_coex300 = 2/sum(1/rho_coex300) ## alternative: compute from average volume
DeltaGPprime300 = DeltaGPprime(Ptrans300,300)*rhomean_coex300
