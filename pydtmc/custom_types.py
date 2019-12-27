# -*- coding: utf-8 -*-

__all__ = [
    # Base
    'ofloat', 'oint',
    'tarray', 'oarray',
    'tgraph', 'ograph', 'tgraphs', 'ographs',
    'tlist_array', 'olist_array', 'tlist_int', 'olist_int', 'tlist_str', 'olist_str',
    'tlists_int', 'olists_int', 'tlists_str', 'olists_str',
    'tnumeric', 'onumeric',
    'tplot', 'oplot',
    # Specific
    'tdict', 'odict', 'tdict_flex', 'odict_flex',
    'tdistributions_flex', 'odistributions_flex',
    'tlist_states', 'olist_states',
    'tstate', 'ostate', 'tstatenames', 'ostatenames', 'tstates', 'ostates', 'tstateswalk', 'ostateswalk', 'tstateswalk_flex', 'ostateswalk_flex',
    'tstatus', 'ostatus',
    'tweights', 'oweights'
]


###########
# IMPORTS #
###########


# Major

import matplotlib.pyplot as _mp
import networkx as _nx
import numpy as _np
import scipy.sparse as _sps

# Minor

from typing import (
    Dict as _Dict,
    Iterable as _Iterable,
    List as _List,
    Optional as _Optional,
    Tuple as _Tuple,
    Union as _Union
)


#########
# TYPES #
#########


# Base

ofloat = _Optional[float]
oint = _Optional[int]

tarray = _np.ndarray
oarray = _Optional[tarray]

tgraph = _nx.DiGraph
ograph = _Optional[tgraph]
tgraphs = _Union[_nx.DiGraph, _nx.MultiDiGraph]
ographs = _Optional[tgraphs]

tlist_array = _List[tarray]
olist_array = _Optional[tlist_array]
tlist_int = _List[int]
olist_int = _Optional[tlist_int]
tlist_str = _List[str]
olist_str = _Optional[tlist_str]

tlists_int = _List[_List[int]]
olists_int = _Optional[tlists_int]
tlists_str = _List[_List[str]]
olists_str = _Optional[tlists_str]

try:
    import pandas as _pd
    tnumeric = _Union[_Iterable, tarray, _sps.spmatrix, _pd.DataFrame, _pd.Series]
except ImportError:
    _pd = None
    tnumeric = _Union[_Iterable, tarray, _sps.spmatrix]

onumeric = _Optional[tnumeric]

tplot = _Tuple[_mp.Figure, _mp.Axes]
oplot = _Optional[tplot]

# Specific

tdict = _Dict[_Tuple[str, str], float]
odict = _Optional[tdict]
tdict_flex = _Dict[_Tuple[str, str], _Union[float, int]]
odict_flex = _Optional[tdict_flex]

tdistributions_flex = _Union[int, _Iterable[tarray]]
odistributions_flex = _Optional[tdistributions_flex]

tlist_states = _Union[_List[int], _List[str]]
olist_states = _Optional[tlist_states]

tstate = _Union[int, str]
ostate = _Optional[tstate]
tstatenames = _Iterable[str]
ostatenames = _Optional[tstatenames]

tstates = _Union[int, str, _Iterable[int], _Iterable[str]]
ostates = _Optional[tstates]
tstateswalk = _Union[_Iterable[int], _Iterable[str]]
ostateswalk = _Optional[tstateswalk]
tstateswalk_flex = _Union[int, _Iterable[int], _Iterable[str]]
ostateswalk_flex = _Optional[tstateswalk_flex]

tstatus = _Union[int, str, tnumeric]
ostatus = _Optional[tstatus]

tweights = _Union[float, int, tnumeric]
oweights = _Optional[tweights]
