#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:57:46 2023
last modified: Sept. 10, 2026
@author: Daniel N. Blaschke

This package implements a new phase transformation kinetics model
"""
from . import data
from . import eos
from . import volumefraction
from . import volumefraction_for
from . import PTkin_figures
from . import utilities

__version__ = '2026.03'
__all__ = ["data", "eos", "volumefraction", "volumefraction_for", "PTkin_figures", "utilities"]
