# -*- coding: utf-8 -*-

__all__ = [
    'color_brightest', 'color_darkest', 'color_gray', 'colors', 'dpi',
    'larray', 'lfloat', 'lint', 'lstr', 'llfloat', 'llint', 'llstr',
    'oarray', 'oaxes', 'odigraph', 'odistribution', 'ofigure', 'ofloat', 'ograph', 'oint', 'oiterable', 'olstr', 'onumeric', 'oplot', 'ostate', 'ostates', 'ostatesflex', 'ostr', 'owalk',
    'tany', 'tarray', 'taxes', 'tdigraph', 'tdistribution', 'tfigure', 'tgraph', 'titerable', 'tnumeric', 'tplot', 'tstate', 'tstates', 'tstatesflex', 'twalk'
]


###########
# IMPORTS #
###########


import matplotlib.pyplot as mlp
import networkx as nx
import numpy as np
import typing as tpg

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import scipy.sparse.csr as spsc
except ImportError:
    spsc = None


############
# SETTINGS #
############


color_brightest = '#FFFFFF'
color_darkest = '#000000'
color_gray = '#E0E0E0'
colors = ['#80B1D3', '#FFED6F', '#B3DE69', '#BEBADA', '#FDB462', '#8DD3C7', '#FB8072', '#FCCDE5']
dpi = 300


#########
# TYPES #
#########


# Base Types
tany = tpg.Any
tarray = np.ndarray
taxes = mlp.Axes
tfigure = mlp.Figure
tgraph = nx.Graph

# List Types
larray = tpg.List[tarray]
lfloat = tpg.List[float]
lint = tpg.List[int]
lstr = tpg.List[str]
llfloat = tpg.List[lfloat]
llint = tpg.List[lint]
llstr = tpg.List[lstr]

# Compound Types
tdigraph = tpg.Union[nx.DiGraph, nx.MultiDiGraph]
tdistribution = tpg.Union[int, larray]
titerable = tpg.Iterable
tplot = tpg.Tuple[tfigure, taxes]
tstate = tpg.Union[int, str]
tstates = tpg.Union[lint, lstr]
tstatesflex = tpg.Union[tstate, tstates]
twalk = tpg.Union[int, tstates]

tnumeric = tpg.Union[titerable, tarray]

if pd:
    tnumeric = tpg.Union[tnumeric, pd.DataFrame, pd.Series]

if spsc:
    tnumeric = tpg.Union[tnumeric, spsc.csr_matrix]

# Optional Types
oarray = tpg.Optional[tarray]
oaxes = tpg.Optional[taxes]
odigraph = tpg.Optional[tdigraph]
odistribution = tpg.Optional[tdistribution]
ofigure = tpg.Optional[tfigure]
ofloat = tpg.Optional[float]
ograph = tpg.Optional[tgraph]
oint = tpg.Optional[int]
oiterable = tpg.Optional[titerable]
olstr = tpg.Optional[lstr]
onumeric = tpg.Optional[tnumeric]
oplot = tpg.Optional[tplot]
ostate = tpg.Optional[tstate]
ostates = tpg.Optional[tstates]
ostatesflex = tpg.Optional[tstatesflex]
ostr = tpg.Optional[str]
owalk = tpg.Optional[twalk]
