#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:57:46 2023
last modified: Nov. 13, 2024
@author: Daniel N. Blaschke

This submodule provides functions to generate various figures
"""
import shutil
import numpy as np
# from numba import jit
# import matplotlib as mpl
# mpl.use('Agg', force=False) # don't need X-window, allow running in a remote terminal session
import matplotlib.pyplot as plt
##### use pdflatex and specify font through preamble:
if shutil.which('latex'):
    # mpl.use("pgf")
    texpreamble = "\n".join([
          r"\usepackage[utf8x]{inputenc}",
          r"\usepackage[T1]{fontenc}",
          r"\DeclareUnicodeCharacter{2212}{-}",
          r"\IfFileExists{fouriernc.sty}",
          r"{\usepackage{fouriernc}}{}",
          r"\usepackage{amsmath}",
    ])
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": texpreamble,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": texpreamble,
    })
##################
fntsize=10
plt.rc('font',**{'family':'serif','size':'10'})
plt.rcParams['font.serif'].insert(0,'Liberation Serif')
plt.rcParams['font.sans-serif'].insert(0,'Liberation Sans')
from matplotlib.ticker import AutoMinorLocator
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
import pandas as pd

def plot_Vfrac_P_Pdot(Vf,P,pdotvals,figtitle,ylabel,extendednamestring="",figsize=(6.5,4),xlimits=None,every=1,legendopts={'loc':'upper left','bbox_to_anchor':(1.01,1), 'handlelength':1.2}):
    '''plot the volume fraction of the 2nd phase as a function of pressure for various pressure rates'''
    ramp = pd.DataFrame(Vf,P).iloc[:,::every]
    pdotvals = np.asarray(pdotvals[::every])
    if np.all(np.abs(pdotvals)<1e-3):
        ramp.columns = pd.Index([rf'{1e6*pi:.0e} GPa/s' for pi in pdotvals])
    else:
        ramp.columns = pd.Index([rf'{pi:.0e} GPa/$\mu$s' for pi in pdotvals])
    ramp.index.name='P [GPa]'
    findx0 = ramp.iloc[:,0]
    findx0 = findx0[findx0<1e-4].index[-1]
    for xi in range(len(pdotvals)):
        finx0 = ramp.iloc[:,xi]
        finx0 = finx0[finx0<1e-4]
        if len(finx0>0):
            findx0 = min(findx0,finx0.index[-1])
        else:
            findx0 = P[0]
    findx1 = ramp.iloc[:,-1]
    findx1 = findx1[findx1>0.9999]
    if len(findx1)>1:
        findx1 = findx1.index[0]
    else:
        findx1 = ramp.iloc[:,-1].index[-1]
    for xi in range(len(pdotvals)):
        finx1 = ramp.iloc[:,xi]
        finx1 = finx1[finx1>0.9999]
        if len(finx1)>1:
            finx1 = finx1.index[0]
        else:
            finx1 = ramp.iloc[:,-1].index[-1]
        findx1 = max(findx1,finx1)
    if xlimits is None:
        xlimits = (findx0,findx1)
    # if xlimits[0]<findx0-3:
    #     xlimits = (findx0-0.5,xlimits[-1])
    # if xlimits[-1]>findx1+3 or xlimits[-1]<xlimits[0]:
    #     xlimits = (xlimits[0],findx1)
    lenpd = int(len(pdotvals)/2)
    pdotvals_pos = pdotvals[:lenpd]
    if min(pdotvals)<0 and len(pdotvals)==2*lenpd and min(pdotvals_pos)>0:
        lnstyle = ['-' for pi in pdotvals_pos] + ['--' for pi in pdotvals_pos]
        if lenpd<=len(colors):
            col = [colors[i] for i in range(lenpd)]
            col += col
        rampfig=ramp.plot(title=figtitle,xlim=xlimits,ylabel=ylabel,figsize=figsize,fontsize=fntsize,style=lnstyle,color=col,ylim=(-0.05,1.05))
    else:
        rampfig=ramp.plot(title=figtitle,xlim=xlimits,ylabel=ylabel,figsize=figsize,fontsize=fntsize,ylim=(-0.05,1.05))
    rampfig.legend(**legendopts,fontsize=fntsize)
    rampfig.xaxis.set_minor_locator(AutoMinorLocator())
    rampfig.yaxis.set_minor_locator(AutoMinorLocator())
    rampfig.get_figure().savefig(f'PTkineticsRamp{extendednamestring}.pdf',format='pdf',bbox_inches='tight')
    plt.close()
    
def plot_relaxtime(tau,taumin,tauplus,pdotvals,figtitle,extendednamestring="",figsize=(4.5,3.5)):
    '''plot the relaxation time'''
    logpdotvals = np.log10(np.abs(pdotvals))
    logtau0 = np.log10([1e3*np.abs(tau[pi]) for pi in pdotvals]) # convert mu-sec to ns and take log10
    logtauplus = np.log10([1e3*np.abs(tauplus[pi]) for pi in pdotvals])
    logtaumin = np.log10([1e3*np.abs(taumin[pi]) for pi in pdotvals])
    relax = pd.DataFrame(np.array([logtau0,logtauplus,logtaumin]).T,logpdotvals)
    relax.index.name = r'$\log_{10}\dot{P}$ [GPa/$\mu$s]'
    relaxfig = relax.plot(title=figtitle,ylabel=r'$\log_{10}\tau$ [ns]',style=['-b','--k',':k'],legend=None,figsize=figsize,fontsize=fntsize)
    relaxfig.xaxis.set_minor_locator(AutoMinorLocator())
    relaxfig.yaxis.set_minor_locator(AutoMinorLocator())
    relaxfig.get_figure().savefig(f'relaxationtimeRamp{extendednamestring}.pdf',format='pdf',bbox_inches='tight')
    plt.close()
    
def plot_onsetP(Ponset,pdotvals,figtitle,extendednamestring="",figsize=(4.5,3.5),linearfit=None,nonlinfit=None,ylimits=None):
    '''plot the onset pressure as a function of strain rate'''
    # approximate pressure at phase transition as fct of loading strain rate P-dot
    Ponset = np.array(Ponset)
    if ylimits is None:
        ylimits = [int(np.nanmin(Ponset)),round(np.nanmax(Ponset)+0.5)]
        if linearfit is not None:
            ylimits[0] = min(ylimits[0],np.nanmin(linearfit))
    lenpd = int(len(pdotvals)/2)
    pdotvals_pos = np.array(pdotvals[:lenpd])
    incinv=False
    styles = ['-b','--k',':k']
    if min(pdotvals)<0 and len(pdotvals)==2*lenpd and min(pdotvals_pos>0):
        incinv=True
        pdotvals = pdotvals_pos
        Ponset_neg = Ponset[lenpd:]
        # PdotP_neg = np.abs(pdotvals)*1e6/Ponset_neg
        # logpdotvals_neg = np.log10(PdotP_neg)
        Ponset = Ponset[:lenpd]
        styles = ['-b','--b','--k',':k']
    PdotP = np.abs(pdotvals)*1e6/Ponset
    logpdotvals = np.log10(PdotP)
    if linearfit is None or nonlinfit is None:
        if incinv:
            ramppressure = [Ponset,Ponset_neg]
            ramppressure = pd.DataFrame(np.array(ramppressure).T,logpdotvals)
        else:
            ramppressure = pd.DataFrame(np.array(Ponset).T,logpdotvals)
    else:
        if incinv:
            ramppressure = [Ponset,Ponset_neg,linearfit,nonlinfit]
        else:
            ramppressure = [Ponset,linearfit,nonlinfit]
        ramppressure = pd.DataFrame(np.array(ramppressure).T,logpdotvals)
    ramppressure.index.name=r'$\log_{10}\dot{P}/P$ [1/s]'
    ramppressurefig = ramppressure.plot(title=figtitle,ylabel=r'P [GPa]',style=styles,legend=None,ylim=ylimits,figsize=figsize,fontsize=fntsize)
    ramppressurefig.xaxis.set_minor_locator(AutoMinorLocator())
    ramppressurefig.yaxis.set_minor_locator(AutoMinorLocator())
    ramppressurefig.get_figure().savefig(f'RampPressure{extendednamestring}.pdf',format='pdf',bbox_inches='tight')
    plt.close()
    
def plot_onsetP2(Ponset,pdotvals,figtitle,extendednamestring="",figsize=(4.5,3.5),ylimits=None,styles = '.b',logscale=False,Ptrans=None):
    '''plot the onset pressure as a function of pressure rate'''
    Ponset = np.array(Ponset)
    if ylimits is None:
        ylimits = [int(np.nanmin(Ponset)),round(np.nanmax(Ponset)+0.5)]
    if logscale:
        pdotvals = np.log10(np.abs(pdotvals))
    if Ptrans is not None:
        ramppressure = pd.DataFrame(np.array([Ponset,np.repeat(Ptrans, len(pdotvals))]).T,pdotvals)
        styles = list(np.array([styles,':k']).flatten())
    else:
        ramppressure = pd.DataFrame(np.array(Ponset).T,pdotvals)
    ramppressure.index.name=r'$\dot{P}$ [GPa/$\mu$s]'
    if logscale:
        ramppressure.index.name=r'$\log_{10}\dot{P}$ [GPa/$\mu$s]'
    ramppressurefig = ramppressure.plot(title=figtitle,ylabel=r'P [GPa]',style=styles,legend=None,ylim=ylimits,figsize=figsize,fontsize=fntsize)
    ramppressurefig.xaxis.set_minor_locator(AutoMinorLocator())
    ramppressurefig.yaxis.set_minor_locator(AutoMinorLocator())
    ramppressurefig.get_figure().savefig(f'OnsetPressure{extendednamestring}.pdf',format='pdf',bbox_inches='tight')
    plt.close()
    
def plot_Vfrac_time_Pdot(Vf,timeP,Ponset,figtitle,ylabel,extendednamestring="",figsize=(6.5,4),slices=None,convertpressurerate=True):
    '''plot the volume fraction of the second phase as a function of time'''
    if np.all(np.abs(timeP)<1e-2):
        timeP = timeP*1e9
        indexname = 'time [ns]'
    else:
        indexname = 'time [s]'
    ramp2 = pd.DataFrame(Vf,timeP)
    ramp2.index.name = indexname
    if convertpressurerate:
        ramp2.columns = (np.asarray(ramp2.columns)*1e6/np.array(Ponset))
        ramp2.columns = [f"{ramp2.columns[i]:.2e} /s" for i in range(len(ramp2.columns))]
    else:
        ramp2.columns = np.asarray(ramp2.columns)*1e6
        ramp2.columns = [f"{ramp2.columns[i]:.2e} GPa/s" for i in range(len(ramp2.columns))]
    if slices is None:
        slices = (0,len(ramp2.iloc[0]),1)
    ramp2 = ramp2.iloc[:,slices[0]:slices[1]:slices[2]]
    if len(ramp2.iloc[0])>4:
        legendopts={'loc':'upper left','bbox_to_anchor':(1.01,1), 'handlelength':1.2}
    else:
        legendopts={'loc':'lower right', 'handlelength':1.2}
    mask1 = ramp2.iloc[:,0].to_numpy()>0
    mask2 = ramp2.iloc[:,-1].to_numpy()<1
    xlimits = (timeP[np.where(mask1)[0][0]],timeP[np.where(mask2)[0][-1]])
    ironrampfig2 = ramp2.plot(title=figtitle,ylabel=ylabel,xlim=xlimits,fontsize=fntsize,figsize=figsize)
    ironrampfig2.legend(**legendopts,fontsize=fntsize)
    ironrampfig2.xaxis.set_minor_locator(AutoMinorLocator())
    ironrampfig2.yaxis.set_minor_locator(AutoMinorLocator())
    ironrampfig2.get_figure().savefig(f'PTkineticsRamp_time{extendednamestring}.pdf',format='pdf',bbox_inches='tight')
    plt.close()
