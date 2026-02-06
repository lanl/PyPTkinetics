#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:57:46 2023
last modified: May 23, 2025
@author: Daniel N. Blaschke
"""
import sys
import os
import numpy as np

dir_path = os.path.realpath(os.path.join(os.path.dirname(__file__),os.pardir))
if dir_path not in sys.path:
    sys.path.append(dir_path)

try:
    from pydislocdyn.metal_data import ISO_c44 as shear
    from pydislocdyn.metal_data import ISO_poisson as poisson
    from pydislocdyn.metal_data import CRC_rho as density
    from pydislocdyn.metal_data import CRC_a as a_lat
    from pydislocdyn.metal_data import CRC_c as c_lat
    from pydislocdyn.metal_data import CRC_T_m as Tm
except ImportError:
    shear = {'Fe':81.6e9, 'Sn':18.4e9}
    poisson = {'Fe':0.293, 'Sn':0.357}
    density = {'Fe':7870, 'Sn':7287}
    a_lat = {'Fe':2.8665e-10, 'Sn':5.8318e-10}
    c_lat = {'Sn':3.1818e-10}
    Tm = {}

## atomic weights taken from the CRC handbook
atommass = {'Al':26.9815386, 'Cu':63.546, 'Fe':55.845, 'Nb':92.90637, 'Au':196.96657, 'Ni':58.693, 'Mo':95.96, 'Ag':107.868,
            'Cd':112.411, 'Mg':24.305, 'Ti':47.867, 'Zn':65.38, 'Zr':91.224, 'Sn':118.710, 'W':183.84, 'Ta':180.94788}
amu = 1.6605390689e-27 ## atomic mass unit [kg]
avogadro = 6.02214076e23 # 1/mol
kB = 1.38064852e-23
e0 = 1.602176634e-19

## default model parameters
kappa = {'Fe':1e3,'Sn':1e3} # m^3 / Js
beta = {'Fe':1e-10, 'Sn':1e-10} # J/m
Ptransition = {'Fe':13, 'Sn':9.4} # GPa
DeltaP = {'Fe':10, 'Sn':10} # GPa
#
gammaAM = {'Fe':50, 'Sn':50} # mJ/m^2 TODO: replace this copper value with iron/tin values!!!
# TODO: for grain sites we need interfacial energy between 2 equal grains gammaAA for k=gammaAA/(2*gammaAM)!!
gammaAA = {key:1.4*item for key, item in gammaAM.items()} # mJ/m^2 TODO: need the correct (average) number, this is just a place holder (inspired by Clemm & Fisher's 1955 choice)
ct = {key:np.sqrt(item/density[key]) for key, item in shear.items()}
# numbers for nucleation rate on dislocations:
rhodis = {'Fe':1e12, 'Sn':1e12} # dislocation density in 1/m^2
burgers = {'Fe':a_lat['Fe']*np.sqrt(3)/2, 'Sn':c_lat['Sn']} ## burgers vector length (for one typical slip system in case of Sn, perhaps take an average over different slip systems here?)
#
graindiameter = {'Fe':1e-2, 'Sn':1e-2} ## average grain diameter D in cm, number suggested in Cahn 1956
grainthickness = {'Fe':1e-8, 'Sn':1e-8} ## average grain boundary thickness delta  in cm, number suggested in Cahn 1956


# TODO: generalize so that we have two sets of B,W for forward and backward trafo (could accept tuples in volfrac code)
greeffB = {'Fe':5e2,'Sn':5e2}
greeffW = {'Fe':5e-2,'Sn':5e-2}
