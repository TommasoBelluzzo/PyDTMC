# -*- coding: utf-8 -*-

__all__ = [
    # Generic
    'ofloat', 'oint',
    'tany', 'titerable',
    # Specific
    'tarray', 'oarray',
    'tbcond', 'obcond',
    'tdists', 'odists',
    'tdists_flex', 'odists_flex',
    'tgraph', 'ograph',
    'tgraphs', 'ographs',
    'tinterval', 'ointerval',
    'tlimit_float', 'olimit_float',
    'tlimit_int', 'olimit_int',
    'tmc', 'omc',
    'tmc_approx', 'omc_approx',
    'tmc_dict', 'omc_dict',
    'tmc_dict_flex', 'omc_dict_flex',
    'tmc_fit', 'omc_fit',
    'tnumeric', 'onumeric',
    'tpart', 'opart',
    'tparts', 'oparts',
    'tplot', 'oplot',
    'trdl', 'ordl',
    'tstate', 'ostate',
    'tstates', 'ostates',
    'tstatus', 'ostatus',
    'ttfunc', 'otfunc',
    'ttimes_in', 'otimes_in',
    'ttimes_out', 'otimes_out',
    'twalk', 'owalk',
    'twalk_flex', 'owalk_flex',
    'tweights', 'oweights',
    # Lists
    'tlist_any', 'olist_any',
    'tlist_array', 'olist_array',
    'tlist_float', 'olist_float',
    'tlist_int', 'olist_int',
    'tlist_str', 'olist_str',
    # Lists of Lists
    'tlists_any', 'olists_any',
    'tlists_array', 'olists_array',
    'tlists_float', 'olists_float',
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

tbcond = Union[float, str]
obcond = Optional[tbcond]

tdists = List[np.ndarray]
odists = Optional[tdists]

tdists_flex = Union[int, tdists]
odists_flex = Optional[tdists_flex]

tgraph = nx.DiGraph
ograph = Optional[tgraph]

tgraphs = Union[nx.DiGraph, nx.MultiDiGraph]
ographs = Optional[tgraphs]

tinterval = Tuple[Union[float, int], Union[float, int]]
ointerval = Optional[tinterval]

tlimit_float = Tuple[float, bool]
olimit_float = Optional[tlimit_float]

tlimit_int = Tuple[int, bool]
olimit_int = Optional[tlimit_int]

tmc = TypeVar('MarkovChain')
omc = Optional[tmc]

tmc_approx = Tuple[tmc, np.ndarray]
omc_approx = Optional[tmc_approx]

tmc_dict = Dict[Tuple[str, str], float]
omc_dict = Optional[tmc_dict]

tmc_dict_flex = Dict[Tuple[str, str], Union[float, int]]
omc_dict_flex = Optional[tmc_dict_flex]

tmc_fit = Tuple[tmc, List[tarray]]
omc_fit = Optional[tmc_fit]

tnumeric = Union[titerable, np.ndarray, spsp.spmatrix, pd.DataFrame, pd.Series] if pd is not None else Union[titerable, tarray, spsp.spmatrix]
onumeric = Optional[tnumeric]

tpart = List[Union[List[int], List[str]]]
opart = Optional[tpart]

tparts = List[tpart]
oparts = Optional[tparts]

tplot = Tuple[pp.Figure, pp.Axes]
oplot = Optional[tplot]

trdl = Tuple[tarray, tarray, tarray]
ordl = Optional[trdl]

tstate = Union[int, str]
ostate = Optional[tstate]

tstates = Union[tstate, List[int], List[str]]
ostates = Optional[tstates]

tstatus = Union[int, str, tnumeric]
ostatus = Optional[tstatus]

ttfunc = Callable[[float, float], float]
otfunc = Optional[ttfunc]

ttimes_in = Union[int, List[int]]
otimes_in = Optional[ttimes_in]

ttimes_out = Union[float, List[float]]
otimes_out = Optional[ttimes_out]

twalk = Union[List[int], List[str]]
owalk = Optional[twalk]

twalk_flex = Union[int, twalk]
owalk_flex = Optional[twalk_flex]

tweights = Union[float, int, tnumeric]
oweights = Optional[tweights]

# Lists

tlist_any = List[tany]
olist_any = Optional[tlist_any]

tlist_array = List[tarray]
olist_array = Optional[tlist_array]

tlist_float = List[float]
olist_float = Optional[tlist_float]

tlist_int = List[int]
olist_int = Optional[tlist_int]

tlist_str = List[str]
olist_str = Optional[tlist_str]

# Lists of Lists

tlists_any = List[tlist_any]
olists_any = Optional[tlists_any]

tlists_array = List[tlist_array]
olists_array = Optional[tlists_array]

tlists_float = List[tlist_float]
olists_float = Optional[tlists_float]

tlists_int = List[tlist_int]
olists_int = Optional[tlists_int]

tlists_str = List[tlist_str]
olists_str = Optional[tlists_str]
