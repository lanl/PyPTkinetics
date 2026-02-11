#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:32:12 2024
last modified: Feb. 10, 2026
@author: dblaschke

This submodule provides various utility functions such as write/read calculation results and loading 3rd party modules if available.
"""
import sys
import os
import multiprocessing
import numpy as np
import pandas as pd

dir_path = os.path.realpath(os.path.join(os.path.dirname(__file__),os.pardir))
if dir_path not in sys.path:
    sys.path.append(dir_path)

nonumba=False
try:
    from numba import jit
except ImportError:
    nonumba=True
    from functools import partial
    def jit(func=None,forceobj=True,nopython=False):
        '''define a dummy decorator if numba is unavailable at runtime'''
        if func is None:
            return partial(jit, forceobj=forceobj,nopython=nopython)
        return func

try:
    from joblib import Parallel, delayed
    Ncpus = multiprocessing.cpu_count()
    ## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
    Ncores = max(1,int(Ncpus/2)) ## don't overcommit, ompthreads=# of threads used by OpenMP subroutines (or 0 if no OpenMP is used)
    ## use half of the available cpus (on systems with hyperthreading this corresponds to the number of physical cpu cores)
except ImportError:
    Ncores = Ncpus = 1 ## must be 1 without joblib

def convert_arg_line_to_args(line):
    '''replacement for according argparse method in order to read our custom log-file format'''
    out = []
    if line[0] != "#":
        currentline = line.lstrip().rstrip().split('=')
        currentline = [i.strip() for i in currentline]
        currentline.insert(1,'=')
        if len(currentline) > 2:
            key = currentline[0]
            if len(currentline)==3 or currentline[3]=='#':
                value = currentline[2]
                if value[-1] == '#':
                    value = value[:-1]
            else:
                value = currentline[2]
                for i in range(len(currentline)-3):
                    addval = currentline[i+3]
                    if addval[0] == '#':
                        break
                    if value[-1] == '#':
                        value = value[:-1]
                        break
                    value += addval
            if key=='material':
                out = [value]
            elif 'None' not in value:
                out = ["--"+key,value]
    return out

def isclose(f1,f2,rtol=1e-05, atol=1e-08):
    '''Returns True if all elements of arrays f1 and f2 are 'close' to one another and their shapes match, and False otherwise.'''
    out = False
    if f1.shape==f2.shape:
        out = np.allclose(f1,f2,equal_nan=True,rtol=rtol,atol=atol)
    return out

def str2bool(arg):
    '''converts a string to bool'''
    if arg in ['True', 'true', '1', 't', 'yes', 'y']:
        out=True
    elif arg in ['False', 'false', '0', 'f', 'no', 'n']:
        out=False
    else:
        raise ValueError(f"cannot convert {arg} to bool")
    return out

def compare_df(f1,f2,rtol=1e-05, atol=1e-08):
    '''Compares two pandas.DataFrames using the pandas.compare method, but ignoring rounding errors (i.e. everything numpy.isclose decides is close enough)'''
    if f1.shape != f2.shape:
        return f"Error: Cannot compare arrays with different shapes: {f1.shape=}, {f2.shape=}"
    if isinstance(f1, np.ndarray) or isinstance(f2, np.ndarray):
        f1 = pd.DataFrame(f1)
        f2 = pd.DataFrame(f2)
    if isinstance(f1, pd.Series) or isinstance(f2, pd.Series):
        themask = pd.Series(np.invert(np.isclose(f1,f2,equal_nan=True,rtol=rtol,atol=atol)),index=f1.index)
    else:
        themask = pd.DataFrame(np.invert(np.isclose(f1,f2,equal_nan=True,rtol=rtol,atol=atol)),index=f1.index,columns=f1.columns)
    f1masked = f1[themask]
    f2masked = f2[themask]
    return f1masked.compare(f2masked)

def writeresults(extendednamestring, pdotvals, pressure, ramppressure, timeP, res, resplus, resmin, tau, tauplus, taumin, optiondict=None):
    '''writes PTkinetic results to disk; use function readresults() to read; if provided, dictionary optiondict will additionally be written to a .log file'''
    kwds = {}
    for xi,pi in enumerate(pdotvals):
        kwds[f'res_{xi}'] = res[pi]
        kwds[f'resplus_{xi}'] = resplus[pi]
        kwds[f'resmin_{xi}'] = resmin[pi]
        kwds[f'tau_{xi}'] = np.array([tau[pi], tauplus[pi], taumin[pi]])
        kwds[f'time_{xi}'] = timeP[pi]
    np.savez_compressed(f"{extendednamestring}_results.npz",pdotvals=np.asarray(pdotvals),pressure=pressure,ramppressure=ramppressure, **kwds)
    if optiondict is not None:
        with open(extendednamestring+".log", "w", encoding="utf8") as logfile:
            logfile.write(f"# {extendednamestring.split('_')[0]}:\n")
            for key, item in optiondict.items():
                if key not in ['verbose', 'skip_calcs', 'Ncores']:
                    logfile.write(f"{key} = {item}\n")
    return 0

def readresults(extendednamestring,wd=None):
    '''reads previously saved PTkinetics results from disk in the current working directory unless keyword wd is set to another folder.'''
    cwd = os.getcwd()
    if wd is not None:
        os.chdir(wd)
    if ".npz" in extendednamestring:
        results = np.load(extendednamestring)
    else:
        results = np.load(f"{extendednamestring}_results.npz")
    pdotvals = results['pdotvals']
    pressure = results['pressure']
    ramppressure = results['ramppressure']
    timeP = {}
    res = {} # volume fraction
    resplus = {}
    resmin = {}
    tau = {} # relaxation time (lower bound, gets closer when pressure resolution is increased above)
    tauplus = {}
    taumin = {}
    for xi,pi in enumerate(pdotvals):
        timeP[pi] = results[f'time_{xi}']
        res[pi] = results[f'res_{xi}']
        resplus[pi] = results[f'resplus_{xi}']
        resmin[pi] = results[f'resmin_{xi}']
        tau[pi], tauplus[pi], taumin[pi] = results[f'tau_{xi}']
    os.chdir(cwd)
    return pdotvals, pressure, ramppressure, timeP, res, resplus, resmin, tau, tauplus, taumin

@jit(nopython=True)
def exp1(x):
    '''this implementation of the exponential integral E1(x) uses approximations 5.1.53 (0<=x<1)
       and 5.1.56 (1<=x<inf) from the Handbook of Mathematical Functions by Abramovitz and Stegun'''
    if 0<=x<1:
        a0,a1,a2,a3,a4,a5 = (-0.57721566,0.99999193,-0.24991055,0.05519968,-0.00976004,0.00107857)
        out = a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 - np.log(x)
    elif x>=1:
        a1,a2,a3,a4 = (8.5733287401,18.0590169730,8.6347608925,0.2677737343)
        b1,b2,b3,b4 = (9.5733223454,25.6329561486,21.0996530827,3.9584969228)
        out = ((x**4+a1*x**3+a2*x**2+a3*x+a4)/(x**4+b1*x**3+b2*x**2+b3*x+b4))*np.exp(-x)/x
    else:
        out = np.nan
    return out

def rampR(x):
    '''computes x*Theta(x)'''
    return x*np.heaviside(x,1/2)

def compare_results(f1,f2,verbose=False,rtol=1e-05, atol=1e-08):
    '''compares results of 2 different runs; can be used for regression testing'''
    success = True
    if isinstance(f1, str):
        f1tuple = readresults(f1)
    elif isinstance(f1, tuple):
        f1tuple = f1
    else:
        raise ValueError("f1 must be a string (filename) or a tuple (output of readresults())")
    if isinstance(f2, str):
        f2tuple = readresults(f2)
    elif isinstance(f2, tuple):
        f2tuple = f2
    else:
        raise ValueError("f2 must be a string (filename) or a tuple (output of readresults())")
    keys = ("pdotvals", "pressure", "ramppressure", "timeP", "res", "resplus", "resmin", "tau", "tauplus", "taumin")
    for i,x in enumerate(keys):
        f1a = f1tuple[i]
        if isinstance(f1a, dict):
            try:
                f1a = pd.DataFrame(f1a)
            except ValueError:
                f1a = pd.Series(f1a)
        f2a = f2tuple[i]
        if isinstance(f2a, dict):
            try:
                f2a = pd.DataFrame(f2a)
            except ValueError:
                f2a = pd.Series(f2a)
        if not isclose(f1a,f2a,rtol=rtol,atol=atol):
            print(f"{x} differs")
            success=False
            if verbose: print(compare_df(f1a,f2a,rtol=rtol,atol=atol))
    return success
