# -*- coding: utf-8 -*-

__all__ = [
    # Generic
    'ofloat', 'oint',
    'tany', 'titerable',
    # Specific
    'tarray', 'oarray',
    'tdistributions', 'odistributions',
    'tgraph', 'ograph',
    'tgraphs', 'ographs',
    'tinterval', 'ointerval',
    'tlimit', 'olimit',
    'tmc', 'omc',
    'tmcdict', 'omcdict',
    'tmcdict_flex', 'omcdict_flex',
    'tnumeric', 'onumeric',
    'tplot', 'oplot',
    'tstate', 'ostate',
    'tstates', 'ostates',
    'tstateswalk', 'ostateswalk',
    'tstateswalk_flex', 'ostateswalk_flex',
    'tstatus', 'ostatus',
    'ttfunc', 'otfunc',
    'tweights', 'oweights',
    # Lists
    'tlist_any', 'olist_any',
    'tlist_array', 'olist_array',
    'tlist_int', 'olist_int',
    'tlist_str', 'olist_str',
    # Lists of Lists
    'tlists_any', 'olists_any',
    'tlists_array', 'olists_array',
    'tlists_int', 'olists_int',
    'tlists_str', 'olists_str'
]


###########
# IMPORTS #
###########


# Major

import matplotlib.pyplot as pp
import networkx as nx
import numpy as np
import scipy.sparse as spsp

try:
    import pandas as pd
except ImportError:
    pd = None

# Minor

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union
)


#########
# TYPES #
#########


# Generic

ofloat = Optional[float]
oint = Optional[int]

tany = Any
titerable = Iterable

# Specific

tarray = np.ndarray
oarray = Optional[tarray]

tdistributions = Union[int, List[tarray]]
odistributions = Optional[tdistributions]

tgraph = nx.DiGraph
ograph = Optional[tgraph]

tgraphs = Union[nx.DiGraph, nx.MultiDiGraph]
ographs = Optional[tgraphs]

tinterval = Tuple[Union[float, int], Union[float, int]]
ointerval = Optional[tinterval]

tlimit = Tuple[int, bool]
olimit = Optional[tlimit]

tmc = TypeVar('MarkovChain')
omc = Optional[tmc]

tmcdict = Dict[Tuple[str, str], float]
omcdict = Optional[tmcdict]

tmcdict_flex = Dict[Tuple[str, str], Union[float, int]]
omcdict_flex = Optional[tmcdict_flex]

tnumeric = Union[titerable, tarray, spsp.spmatrix, pd.DataFrame, pd.Series] if pd is not None else Union[titerable, tarray, spsp.spmatrix]
onumeric = Optional[tnumeric]

tplot = Tuple[pp.Figure, pp.Axes]
oplot = Optional[tplot]

tstate = Union[int, str]
ostate = Optional[tstate]

tstates = Union[tstate, List[int], List[str]]
ostates = Optional[tstates]

tstateswalk = Union[List[int], List[str]]
ostateswalk = Optional[tstateswalk]

tstateswalk_flex = Union[int, List[int], List[str]]
ostateswalk_flex = Optional[tstateswalk_flex]

tstatus = Union[int, str, tnumeric]
ostatus = Optional[tstatus]

ttfunc = Callable[[float, float], float]
otfunc = Optional[ttfunc]

tweights = Union[float, int, tnumeric]
oweights = Optional[tweights]

# Lists

tlist_any = List[tany]
olist_any = Optional[tlist_any]

tlist_array = List[tarray]
olist_array = Optional[tlist_array]

tlist_int = List[int]
olist_int = Optional[tlist_int]

tlist_str = List[str]
olist_str = Optional[tlist_str]

# Lists of Lists

tlists_any = List[tlist_any]
olists_any = Optional[tlists_any]

tlists_array = List[tlist_array]
olists_array = Optional[tlists_array]

tlists_int = List[tlist_int]
olists_int = Optional[tlists_int]

tlists_str = List[tlist_str]
olists_str = Optional[tlists_str]
