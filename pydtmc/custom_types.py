# -*- coding: utf-8 -*-

__all__ = [
    # Generic
    'ofloat', 'oint', 'ostr',
    'tany', 'texception', 'titerable',
    'tarray', 'oarray',
    'tcache', 'ocache',
    'tgraph', 'ograph',
    'tgraphs', 'ographs',
    'tfile', 'ofile',
    'tlimit_float', 'olimit_float',
    'tlimit_int', 'olimit_int',
    'tmc', 'omc',
    'tplot', 'oplot',
    'trand', 'orand',
    'tnumeric', 'onumeric',
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
    'tlists_str', 'olists_str',
    # Specific
    'tbcond', 'obcond',
    'tdists_flex', 'odists_flex',
    'tfitres', 'ofitres',
    'tgenres', 'ogenres',
    'tgenres_ext', 'ogenres_ext',
    'tinterval', 'ointerval',
    'tmc_dict', 'omc_dict',
    'tmc_dict_flex', 'omc_dict_flex',
    'tpart', 'opart',
    'tparts', 'oparts',
    'trdl', 'ordl',
    'tredists', 'oredists',
    'tstate', 'ostate',
    'tstates', 'ostates',
    'tstatus', 'ostatus',
    'ttfunc', 'otfunc',
    'ttimes_in', 'otimes_in',
    'ttimes_out', 'otimes_out',
    'twalk', 'owalk',
    'twalk_flex', 'owalk_flex',
    'tweights', 'oweights'
]


###########
# IMPORTS #
###########

# Standard

# noinspection PyPep8Naming
from typing import (
    Any as _Any,
    Callable as _Callable,
    Dict as _Dict,
    Iterable as _Iterable,
    List as _List,
    Optional as _Optional,
    Tuple as _Tuple,
    TypeVar as _TypeVar,
    Union as _Union
)

# Libraries

import matplotlib.pyplot as _mplp
import networkx as _nx
import numpy as _np
import numpy.random as _npr
import scipy.sparse as _spsp

try:
    import pandas as _pd
except ImportError:  # noqa
    _pd = None


#########
# TYPES #
#########

# Generic

ofloat = _Optional[float]
oint = _Optional[int]
ostr = _Optional[str]

tany = _Any
texception = Exception
titerable = _Iterable

tarray = _np.ndarray
oarray = _Optional[tarray]

tcache = _Dict[str, tany]
ocache = _Optional[tcache]

tfile = _Tuple[str, str]
ofile = _Optional[tfile]

tgraph = _nx.DiGraph
ograph = _Optional[tgraph]

tgraphs = _Union[tgraph, _nx.MultiDiGraph]
ographs = _Optional[tgraphs]

tlimit_float = _Tuple[float, bool]
olimit_float = _Optional[tlimit_float]

tlimit_int = _Tuple[int, bool]
olimit_int = _Optional[tlimit_int]

tmc = _TypeVar('MarkovChain')
omc = _Optional[tmc]

tplot = _Tuple[_mplp.Figure, _mplp.Axes]
oplot = _Optional[tplot]

tnumeric = _Union[_np.ndarray, _spsp.spmatrix] if _pd is None else _Union[_np.ndarray, _spsp.spmatrix, _pd.DataFrame, _pd.Series]
onumeric = _Optional[tnumeric]

trand = _npr.RandomState
orand = _Optional[trand]

# Lists

tlist_any = _List[tany]
olist_any = _Optional[tlist_any]

tlist_array = _List[tarray]
olist_array = _Optional[tlist_array]

tlist_float = _List[float]
olist_float = _Optional[tlist_float]

tlist_int = _List[int]
olist_int = _Optional[tlist_int]

tlist_str = _List[str]
olist_str = _Optional[tlist_str]

# Lists of Lists

tlists_any = _List[tlist_any]
olists_any = _Optional[tlists_any]

tlists_array = _List[tlist_array]
olists_array = _Optional[tlists_array]

tlists_float = _List[tlist_float]
olists_float = _Optional[tlists_float]

tlists_int = _List[tlist_int]
olists_int = _Optional[tlists_int]

tlists_str = _List[tlist_str]
olists_str = _Optional[tlists_str]

# Specific

tbcond = _Union[float, int, str]
obcond = _Optional[tbcond]

tdists_flex = _Union[int, tlist_array]
odists_flex = _Optional[tdists_flex]

tfitres = _Tuple[oarray, ostr]
ofitres = _Optional[tfitres]

tgenres = _Tuple[oarray, ostr]
ogenres = _Optional[tgenres]

tgenres_ext = _Tuple[oarray, olist_str, ostr]
ogenres_ext = _Optional[tgenres_ext]

tinterval = _Tuple[_Union[float, int], _Union[float, int]]
ointerval = _Optional[tinterval]

tmc_dict = _Dict[_Tuple[str, str], float]
omc_dict = _Optional[tmc_dict]

tmc_dict_flex = _Dict[_Tuple[str, str], _Union[float, int]]
omc_dict_flex = _Optional[tmc_dict_flex]

tpart = _List[_Union[tlist_int, tlist_str]]
opart = _Optional[tpart]

tparts = _List[tpart]
oparts = _Optional[tparts]

trdl = _Tuple[tarray, tarray, tarray]
ordl = _Optional[trdl]

tredists = _Union[tarray, tlist_array]
oredists = _Optional[tredists]

tstate = _Union[int, str]
ostate = _Optional[tstate]

tstates = _Union[tstate, tlist_int, tlist_str]
ostates = _Optional[tstates]

tstatus = _Union[int, str, tnumeric]
ostatus = _Optional[tstatus]

ttfunc = _Callable[[int, float, int, float], float]
otfunc = _Optional[ttfunc]

ttimes_in = _Union[int, tlist_int]
otimes_in = _Optional[ttimes_in]

ttimes_out = _Union[float, tlist_float]
otimes_out = _Optional[ttimes_out]

twalk = _Union[tlist_int, tlist_str]
owalk = _Optional[twalk]

twalk_flex = _Union[int, twalk]
owalk_flex = _Optional[twalk_flex]

tweights = _Union[float, int, tnumeric]
oweights = _Optional[tweights]
