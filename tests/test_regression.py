#!/usr/bin/env python3
# test suite for PyPTkinetics
# Author: Daniel N. Blaschke
# Copyright (c) 2025, Triad National Security, LLC. All rights reserved.
# Date: Feb. 10, 2026
'''This script implements regression-testing for PyPTkinetics and is meant to be called by pytest.'''
import os
import sys
import subprocess
import pathlib
# import numpy as np
# import sympy as sp
dir_path = str(pathlib.Path(__file__).resolve().parents[1])
if dir_path not in sys.path:
    sys.path.append(dir_path)
testpath = pathlib.Path(__file__).resolve().parents[0]
dir_path = pathlib.Path(__file__).resolve().parents[1]

import PTkinetics as ptk

def runscript(scriptname,args,logfname):
    '''Run script "scriptname" as a subprocess passing a list of command line arguments "args" and saving its stdout to a file "logfname"'''
    out = -1
    with open(logfname, 'w', encoding="utf8") as logfile:
        command = [pathlib.Path(dir_path,scriptname)]
        if sys.platform=='win32':
            command = ["python",pathlib.Path(dir_path,scriptname)]
        with subprocess.Popen(command+args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
            for line in subproc.stdout:
                sys.stdout.write(line)
                logfile.write(line)
            subproc.wait()
            out = subproc.returncode
    return out

def test_regression():
    '''perform a number of regression tests'''
    commandargs = "Fe -v --Npdot=4 --resolution=1000".split(" ")
    os.chdir(testpath)
    ## run microstructure dependent model with low resolution and small number of pressure rates and assert results
    assert runscript(dir_path/"PTkinetics"/"ramploading.py",commandargs,'run_test1.log')==0
    assert ptk.utilities.compare_results(str(testpath/"Iron_baseline.npz"),str(testpath/"Iron_d1e+12_g0.01gb20ge0gc0_results.npz"),verbose=True)
    ## run greeff model with same low res. etc., but this time read all commanline args from a log file, then assert results
    assert runscript(dir_path/"PTkinetics"/"ramploading.py",["@Iron_baseline_greeff.log"],'run_test2.log')==0
    assert ptk.utilities.compare_results(str(testpath/"Iron_baseline_greeff.npz"),str(testpath/"Iron_greeff_B=5e+02_W=5e-02_results.npz"),verbose=True)
