#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:57:46 2023
last modified: Feb. 6, 2026
@author: Daniel N. Blaschke

This script applies the phase transformation kinetics models implemented in this package
to ramp loading examples.
"""
import sys
import os
import argparse
import numpy as np

dir_path = os.path.realpath(os.path.join(os.path.dirname(__file__),os.pardir))
if dir_path not in sys.path:
    sys.path.append(dir_path)

from PTkinetics import data
from PTkinetics.volumefraction_for import compute_prefactors, relaxtime, t_of_P_ramp, figrain, sigrain, lambdaE_hd, lambdaE_grain
from PTkinetics.PTkin_figures import plot_Vfrac_P_Pdot, plot_relaxtime, plot_onsetP, plot_onsetP2 #, plot_Vfrac_time_Pdot
from PTkinetics.utilities import Ncores, Ncpus, nonumba, writeresults, readresults, str2bool, convert_arg_line_to_args#, rampR
if Ncpus>1:
    from PTkinetics.utilities import Parallel, delayed
else:
    print("Warning: Parallelization is disabled because 'joblib' is not installed")

implemented = ['Fe']
parser = argparse.ArgumentParser(usage=f"\n{sys.argv[0]} <options> <material>",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args
parser.add_argument('material', type=str, help=f'material we are caluclating for; choose between {implemented}')
parser.add_argument('-include_hom','--include_hom',type=str2bool,default=False,help='homogeneous nucleation is not the driving mechanism (does not match experiments if active)')
parser.add_argument('-include_disloc','--include_disloc',type=str2bool,default=True,help='nucleation on dislocations')
parser.add_argument('-include_grains','--include_grains',type=str2bool,default=True,help='nucleation on grains')
parser.add_argument('-graindiameter','--graindiameter',type=float,default=None,help='average grain diameter D in cm, defaults to reading from submodule "data" for the chosen material')
parser.add_argument('-grainthickness','--grainthickness',type=float,default=None,help='average grain boundary thickness delta in cm, defaults to reading from submodule "data" for the chosen material')
parser.add_argument('-include_gb','--include_gb',type=str2bool,default=True,help="nucleation on grain boundaries")
parser.add_argument('-include_ge','--include_ge',type=str2bool,default=False,help="nucleation on grain edges (quite slow)")
parser.add_argument('-include_gc','--include_gc',type=str2bool,default=False,help="nucleation on grain corners (subleading)")
parser.add_argument('-maxP','--maxP',type=float,default=0,help='overrides the maximum pressure to probe if this is >0')
parser.add_argument('-gammaAM','--gammaAM',type=float,default=None,help='average interfacial energy in mJ/m^2 between grains of different phases')
parser.add_argument('-gammaAA','--gammaAA',type=float,default=None,help='average interfacial energy in mJ/m^2 between grains of the same (initial) phase')
parser.add_argument('-rhodis','--rhodis',type=float,default=None,help='dislocation density in 1/m^2')
parser.add_argument('-DeltaP','--DeltaP',type=float,default=None,help='difference in pressure between the coexistyence curve and the spinodal in GPa')
parser.add_argument('-kappa','--kappa',type=float,default=None,help='model parameter in m^3 / Js')
parser.add_argument('-beta','--beta',type=float,default=None,help='model parameter in J/m')
parser.add_argument('-resolution','--resolution',type=int,default=10000,help='resolution in pressure used for the calculations')
parser.add_argument('-v','--verbose',action='store_true')
parser.add_argument('-Npdot','--Npdot',type=int,default=16,help='choose between: 4, 6, 7, 8, 10, 16, 18 to select one of the pre-defined lists of pressure rates to consider')
parser.add_argument('-Ncores','--Ncores',type=int,default=Ncores,help='override number of threads to use for parallelization')
parser.add_argument('-skip_calcs','--skip_calcs',action='store_true',help='attempt to read from the cwd on disk previously calculated results to plot')
parser.add_argument('-include_inverse','--include_inverse',type=str2bool,default=True,help="include also the inverse phase transformation")
parser.add_argument('-showfigs','--showfigs',type=str2bool,default=False,help='show the generated figures in addition to saving them (useful when running in jupyter)')

if __name__ == '__main__':
    opts = parser.parse_args()
    metal = opts.material
    if metal == 'Fe':
        from PTkinetics.eos.iron import Ptrans300, rhomean_coex300, DeltaGPprime300
        extendednamestring = "Iron_for"
        figtitle = r'$\alpha$(bcc)$\to\epsilon$(hcp) transition in Fe'
        figtitle_inv = r'$\alpha$(bcc)$\to\epsilon$(hcp)$\to\alpha$(bcc) transition in Fe'
        ylabel = r'$\epsilon$-Fe volume fraction'
    # elif metal == 'Sn':
    #     from PTkinetics.eos.tin import Ptrans300, rhomean_coex300, DeltaGPprime300
    #     extendednamestring = "Tin_for"
    #     figtitle = r'$\beta\to\gamma$ transition in Sn'
    #     figtitle_inv = r'$\beta\to\gamma\to\beta$ transition in Sn'
    #     ylabel = r'$\gamma$-Sn volume fraction'
    else:
        raise ValueError(f"not implemented for {metal=}; \nplease choose one of: {implemented}")
    ## load default model parameters of new PT kinetics model:
    atommass = data.atommass[metal]
    if opts.graindiameter is None:
        opts.graindiameter = data.graindiameter[metal]
    if opts.grainthickness is None:
        opts.grainthickness = data.grainthickness[metal]
    if opts.kappa is None:
        opts.kappa = data.kappa[metal]
    if opts.beta is None:
        opts.beta = data.beta[metal]
    # Ptransition = data.Ptransition[metal] ## use value calculated from eos below instead
    if opts.DeltaP is None:
        opts.DeltaP = data.DeltaP[metal]
    if opts.gammaAM is None:
        opts.gammaAM = data.gammaAM[metal]
    if opts.gammaAA is None:
        opts.gammaAA = data.gammaAA[metal]
    if opts.rhodis is None:
        opts.rhodis = data.rhodis[metal]
    burgers = data.burgers[metal]
    shear = data.shear[metal]/1e9 ## convert to GPa
    poisson = data.poisson[metal]

    if Ncpus == 1 and Ncores > 1:
        raise ValueError(f"{Ncores=} requested by user, but parallelization is unvailable; please install joblib")

    ## calculate and make importable all pressure-independent stuff:
    Ttarget = 300 ## if we change this, we need to compute the 3 quantities below for the according T
    Ptransition = Ptrans300/1e9 # GPa
    rpref,epshom,Ndotpref,rhob2,alpha_dis = compute_prefactors(deltagpprime=DeltaGPprime300,rhomean=rhomean_coex300,gammaam=opts.gammaAM,rhodis=opts.rhodis,burgers=burgers,
                                                               shear=shear,poisson=poisson,atommass=atommass,deltap=opts.DeltaP,kappa=opts.kappa,beta=opts.beta)
    figtitle_pos = figtitle
    if opts.include_inverse:
        figtitle = figtitle_inv
        xmin = 0.1
    else:
        xmin = round(Ptransition-0.5)
    
    def lambdaE(t,Pdot):
        '''volume fraction from hom. nucleation and nucl. on dislocations'''
        return lambdaE_hd(t=t, Pdot=Pdot, rpref=rpref, Ndotpref=Ndotpref, epshom=epshom,rhob2=rhob2,alpha_dis=alpha_dis,Ttarget=Ttarget,include_hom=opts.include_hom, include_disloc=opts.include_disloc)
    
    #### nucleation at grain boundaries (2), edges (1), and corners (0)
    k = 0.5*opts.gammaAA/opts.gammaAM
    f2g, f1g, f0g = figrain(k,opts.include_gb,opts.include_ge,opts.include_gc)
    
    if metal in ['Fe', 'Sn']:
        ## constants specific to the bcc lattice as well as for beta tin which has a body-centered tetragonal lattice with c/a<sqrt(2), therefore its Voronai polyhedra are truncated octahedra:
        s2, s1, s0 = sigrain('trunc_oct')
    else:
        raise ValueError(f"not implemented for {metal=}")
    
    def lambdaEgrain(t,Pdot,delta,D):
        '''volume fraction from nucleation on grain boundaries, edges, and corners'''
        return lambdaE_grain(t=t, Pdot=Pdot,delta=delta,D=D,rpref=rpref,Ndotpref=Ndotpref,epshom=epshom,f2g=f2g,f1g=f1g,f0g=f0g,s2=s2,s1=s1,s0=s0,Ttarget=Ttarget)

    if opts.Npdot==4:
        pdotvals = [1,10,100,1000]
    elif opts.Npdot==6:
        pdotvals = [1e-6,5e-6,1e-5,5e-5,1e-4,5e-4]
    elif opts.Npdot==7:
        pdotvals = [1,5,10,50,100,500,1000]
    elif opts.Npdot==8:
        pdotvals = [1,10,100,1000,1e4,1e5,1e6,1e7]
    elif opts.Npdot==10:
        pdotvals = [1e-8,5e-8,1e-7,5e-7,1e-6,5e-6,1e-5,5e-5,1e-4,5e-4]
    elif opts.Npdot==18:
        pdotvals = [1e-1,1,5,10,50,100,500,1000,5e3,1e4,5e4,1e5,5e5,1e6,5e6,1e7,5e7,1e8]
    elif opts.Npdot==16:
        pdotvals = [1e-6,1e-3,1e-1,1,5,10,50,100,500,1000,5e3,1e4,5e4,1e5,5e5,1e6] ## exp. can't reach rates above 1e5 at the moment, add 1e6 to compare to future exp.
    else:
        raise ValueError(f"{opts.Npdot=} not implemented")
    if np.max(pdotvals)<=1:
        pressure = np.exp(np.linspace(np.log(xmin),np.log(round(Ptransition)+5),opts.resolution)) # GPa
    elif np.max(pdotvals)<5e3:
        pressure = np.exp(np.linspace(np.log(xmin),np.log(round(Ptransition)+15),opts.resolution)) # GPa
    elif np.max(pdotvals)<5e6:
        pressure = np.exp(np.linspace(np.log(xmin),np.log(round(Ptransition)+23),opts.resolution)) # GPa
    else:
        pressure = np.exp(np.linspace(np.log(xmin),np.log(round(Ptransition)+33),opts.resolution)) # GPa
    if opts.maxP>0:
        pressure = np.exp(np.linspace(np.log(pressure[0]),np.log(opts.maxP),opts.resolution)) # GPa
    if opts.include_inverse:
        pdotvals = pdotvals + [-pi for pi in pdotvals]
    onsetpressure = np.zeros(np.shape(pdotvals))
    timeP = {}
    res = {} # volume fraction
    resplus = {}
    resmin = {}
    tau = {} # relaxation time (lower bound, gets closer when pressure resolution is increased above)
    tauplus = {}
    taumin = {}
    
    if opts.include_hom:
        extendednamestring += f"_h{opts.gammaAM}"
    if opts.include_disloc:
        extendednamestring += f"_d{opts.rhodis:.0e}"
    if opts.include_grains:
        extendednamestring += f"_g{opts.graindiameter}gb{100*f2g:.0f}ge{100*f1g:.0f}gc{100*f0g:.0f}"
    
    def maincomputations(xi,pi):
        if opts.verbose:
            print(f"calculating for Pdot={pi:.2e} ...")
        ## Note: if pi<0, volfrac and onsetpressure change their meaning and are for volfrac of 1st phase (inverse phase trafo and t grows as P shrinks from Ptrans)
        timeP = t_of_P_ramp(pressure,pi,Ptransition)
        lamE = lambdaE(timeP,abs(pi))
        if opts.include_grains:
            lamE += np.asarray([lambdaEgrain(timeP[ti],abs(pi), delta=opts.grainthickness, D=opts.graindiameter) for ti in range(len(timeP))])
        volfrac = (1-np.exp(-lamE))
        resplus = 1-np.exp(-lamE*1e3)
        resmin = 1-np.exp(-lamE/1e3)
        tau = relaxtime(volfrac, timeP)
        tauplus = relaxtime(resplus, timeP)
        taumin = relaxtime(resmin, timeP)
        if len(pressure[volfrac<=0.05])>0:
            onsetpressure = pressure[volfrac<=0.05][-1] ## pressure when volume fraction reaches 5%
        elif len(pressure)>len(pressure[volfrac>0.05])>0:
            onsetpressure = pressure[volfrac>0.05][0]
        else:
            onsetpressure = None
        if pi>0 and onsetpressure is not None and np.isclose(onsetpressure,pressure[-1]):
            print(f"Warning: {onsetpressure=}, {pressure[-1]=}, expect clipping for Pdot={pi:.2e}")
        return (volfrac, resplus, resmin, tau, tauplus, taumin, onsetpressure, timeP)

    if opts.skip_calcs:
        pdotvals, pressure, onsetpressure, timeP, res, resplus, resmin, tau, tauplus, taumin = readresults(extendednamestring)
    else:
        if Ncores > 1:
            results = Parallel(n_jobs=Ncores)(delayed(maincomputations)(xi, pi) for xi,pi in enumerate(pdotvals))
        else:
            results = [maincomputations(xi, pi) for xi,pi in enumerate(pdotvals)]
        
        for xi,pi in enumerate(pdotvals):
            res[pi], resplus[pi], resmin[pi], tau[pi], tauplus[pi], taumin[pi], onsetpressure[xi], timeP[pi] = results[xi]
            if pi<0:
                res[pi] = 1-res[pi]
                resplus[pi] = 1-resplus[pi]
                resmin[pi] = 1-resmin[pi]
                if len(onsetpressure_tmp := pressure[res[pi]<=0.95])>0:
                    onsetpressure[xi] = onsetpressure_tmp[-1]
                else:
                    onsetpressure[xi] = np.min(pressure)
                    print(f"Warning: {onsetpressure[xi]=}, {min(pressure)=}, expect clipping for Pdot={pi:.2e}")
        
        optiondict = vars(opts)
        if (maxP:=int(onsetpressure[-1]+1))<int(pressure[-1]+1):
            optiondict |= {'# recommended maxP:\n#maxP':maxP}
        writeresults(extendednamestring, pdotvals, pressure, onsetpressure, timeP, res, resplus, resmin, tau, tauplus, taumin, optiondict)
    
    select_pos = np.array(pdotvals)>0
    pdotvals_pos = np.array(pdotvals)[select_pos]#[pi for pi in pdotvals if pi>0] # plot only for positive pressure rates (i.e. forward trafo)
    onsetpressure_pos = onsetpressure[select_pos]
    if len(pdotvals_pos) > 7:
        figsize = (9,3.5)
        # xlimits = (xmin,xmin+15.5)
        every = 2
    else:
        figsize = (6.5,4)
        # xlimits = (xmin,xmin+4)
        every = 1
    # if np.max(pdotvals) <= 1 or include_inverse:
    xlimits = None
    if onsetpressure is not None:
        plot_Vfrac_P_Pdot(res,pressure,pdotvals,figtitle=figtitle,ylabel=ylabel,extendednamestring=extendednamestring,figsize=figsize,xlimits=xlimits,every=every,showfig=opts.showfigs)
        
    # relaxation time:
    skiptau = 5e6
    if len(pdotvals_pos) < 8:
        startat = 0
    else:
        startat = 2
    pdotvals2 = np.asarray(pdotvals_pos)
    pdotvals2 = (pdotvals2[pdotvals2<skiptau])[startat:]
    plot_relaxtime(tau=tau, taumin=taumin, tauplus=tauplus, pdotvals=pdotvals2, extendednamestring=extendednamestring,figtitle=figtitle_pos,showfig=opts.showfigs)
    
    # approximate pressure at phase transition as fct of loading strain rate P-dot
    PdotP = (np.abs(pdotvals_pos)*1e6/np.array(onsetpressure_pos))
    logpdotvals = np.log10(PdotP)
    if metal == 'Fe':
        PdotP[PdotP<5e5] = np.nan
        logpdotvals[logpdotvals<2.] = np.nan
        logpdotvals[logpdotvals>7.] = np.nan
        linearfit = 10.8+0.55*logpdotvals
        nonlinfit = 1.15*(PdotP)**0.18
    else:
        linearfit = None
        nonlinfit = None
    if onsetpressure is not None and np.isclose(np.max(pressure),np.max(onsetpressure)):
        print(f"Warning: {max(onsetpressure)=} reached maximum probed pressure ({max(pressure)=}); may want to increase the latter to avoid clipping in the onsetP plot?")
    plot_onsetP(onsetpressure_pos,pdotvals=pdotvals_pos,figtitle=figtitle_pos,extendednamestring=extendednamestring,linearfit=linearfit,nonlinfit=nonlinfit,showfig=opts.showfigs)
    plot_onsetP2(onsetpressure,pdotvals=pdotvals,figtitle=figtitle,extendednamestring=extendednamestring,Ptrans=Ptransition,showfig=opts.showfigs)
