#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:57:46 2023
last modified: July 18, 2024
@author: Daniel N. Blaschke

EOS for alpha and epsilon iron according to Dean's notes and Boettger & Wallace' paper; BUT everything (except for very few model parameters) is in SI units
"""

import numpy as np
# from numba import jit
from scipy import optimize
from ..data import atommass, avogadro, kB #, amu

# all tuples contain values for beta phase and gamma phase (in this order):
phases = ('alpha','epsilon')
## values for [alpha,epsilon] iron (according to Dean's notes and Boettger & Wallace' paper):
Vstar = np.array([7.0047,6.5984]) # cm^3/mol
V300 = np.array([7.093,6.73]) # cm^3/mol;
Bstar = 1e9*np.array([176.64,181.5]) # Pa
B1star = np.array([4.7041,5.74])
PH300 = 1e9*np.array([2.1,3.3]) # Pa
Phistar = np.array([0,5533]) # J/mol
Theta0300 = np.array([301,261]) # K
Theta2300 = np.array([420,364]) # K
gamma0300 = np.array([1.82,2.8])
Tm = 1135

## convert everything to SI units:
convert_mol2kg = atommass['Fe']*1e-3 ## amu*avogadro = 1e-3 by definition
rhostar = 1e6*convert_mol2kg/Vstar ## kg/m^3
rho300 = 1e6*convert_mol2kg/V300
Phistar_SI = Phistar/convert_mol2kg # J/kg

def phasearray(arg):
    '''ensure arg is a 2d array and turn it into one otherwise'''
    if isinstance(arg,np.ndarray) and len(arg)==1:
        arg = np.array([arg[0],arg[0]])
    if isinstance(arg,float):
        arg = np.array([arg,arg])
    return arg


def Theta0(rho,rho300=rho300,Theta0300=Theta0300,gamma0300=gamma0300):
    '''subroutine of FH() and Pressure()'''
    return Theta0300*np.exp(gamma0300*(1-rho300/rho))

def Theta2(rho,rho300=rho300,Theta2300=Theta2300,gamma0300=gamma0300):
    '''subroutine of FH() and Pressure()'''
    return Theta2300*np.exp(gamma0300*(1-rho300/rho))

# @jit(nopython=True)
def Pressure(rho,T):
    '''returns pressure as a function of density and temperature for both phases'''
    rho = phasearray(rho)
    eta = (3/2)*(B1star-1)*(np.cbrt(rhostar/rho)-1)
    Pphi = -(2*Bstar/(B1star-1))*eta*np.exp(-eta)*np.cbrt((rho/rhostar)**2)
    PH = 3e6*avogadro*kB*T*gamma0300*(1+(Theta2(rho,rho300,Theta2300,gamma0300)/T)**2/20)/V300
    return Pphi+PH

def Density(P,T):
    '''returns density as a function of pressure and temperature for both phases'''
    out = np.zeros((2))
    for i in range(2):
        def f(x):
            return abs(Pressure(x,T)[i]-P)
        rho = optimize.fsolve(f, rhostar[i])
        out[i] = rho[0]
    return out

def FPhi0(rho):
    '''returns the static lattice contribution to the Helmholtz free energy in J/kg'''
    eta = (3/2)*(B1star-1)*(np.cbrt(rhostar/rho)-1)
    return Phistar_SI+4*Bstar*(1-(1+eta)*np.exp(-eta))/(B1star-1)**2/rhostar

def FH(rho,T):
    '''returns the quasiharmonic free energy in J/kg'''
    return 3*avogadro*kB*T*(-np.log(T/Theta0(rho,rho300,Theta0300,gamma0300)) + (Theta2(rho,rho300,Theta2300,gamma0300)/T)**2/40)/convert_mol2kg

def Fmag(T):
    '''returns the magnetic free energy for alpha iron (and zero for epsilon iron) in J/kg'''
    return np.array([4680*((1-T/Tm)*np.log((1+np.sqrt(T/Tm))/(1-np.sqrt(T/Tm))) - 2*np.sqrt(T/Tm) + (4/3)*(T/Tm)**(3/2))/convert_mol2kg,0])

def Fcond(rho,T):
    '''returns the electron free energy in J/kg'''
    NAGammaV300 = 2.5e-3 # J/mol/K^2 for both phases according to Boettger & Wallace
    return -0.5*NAGammaV300*(rho300/rho)**1.3*T**2/convert_mol2kg

def Ffree(rho,T):
    '''returns the Helmholtz free energy as a function of density and temperature in J/kg'''
    return FPhi0(rho) + FH(rho, T) + Fmag(T) + Fcond(rho, T)

def Gibbs(P,T):
    '''returns the Gibbs free energy as a function of pressure and temperature for both phases'''
    out = np.zeros(2)
    rho = Density(P,T)
    if isinstance(P,np.ndarray) and len(P)==1:
        P = P[0]
    for i in range(2):
        out[i] = Ffree(rho,T)[i] + P/rho[i]
    return out

def DeltaGibbs(P,T):
    '''returns the difference in Gibbs free energy of the two phases (alpha and epsilon iron) as a function of pressure and temperature'''
    G = Gibbs(P,T)
    return G[0]-G[1]

def Pcoex(T):
    '''determines the transition pressure between the two phases (alpha and epsilon iron) as a function of temperature'''
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
