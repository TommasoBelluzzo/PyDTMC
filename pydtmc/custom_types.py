# -*- coding: utf-8 -*-

__all__ = [
    # Generic
    'obool', 'ofloat', 'oint', 'ostr',
    'tscalar', 'oscalar',
    'tany', 'texception', 'titerable',
    'tarray', 'oarray',
    'tcache', 'ocache',
    'tgraph', 'ograph',
    'tgraphs', 'ographs',
    'tfile', 'ofile',
    'tlimit_float', 'olimit_float',
    'tlimit_int', 'olimit_int',
    'tlimit_scalar', 'olimit_scalar',
    'tmc', 'omc',
    'tplot', 'oplot',
    'trand', 'orand',
    'trandfunc', 'orandfunc',
    'trandfunc_flex', 'orandfunc_flex',
    'tnumeric', 'onumeric',
    # Lists
    'tlist_any', 'olist_any',
    'tlist_array', 'olist_array',
    'tlist_float', 'olist_float',
    'tlist_int', 'olist_int',
    'tlist_mc', 'olist_mc',
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
    'ttest', 'otest',
    'ttest_chi2', 'otest_chi2',
    'ttfunc', 'otfunc',
    'ttimes_in', 'otimes_in',
    'ttimes_out', 'otimes_out',
    'tvalid_states', 'ovalid_states',
    'tvalid_walk', 'ovalid_walk',
    'tvalid_walks', 'ovalid_walks',
    'twalk', 'owalk',
    'twalk_flex', 'owalk_flex',
    'twalks', 'owalks',
    'tweights', 'oweights'
]


###########
# IMPORTS #
###########

# Standard

from typing import (
    Any as _tp_Any,
    Callable as _tp_Callable,
    Dict as _tp_Dict,
    Iterable as _tp_Iterable,
    List as _tp_List,
    Optional as _tp_Optional,
    Tuple as _tp_Tuple,
    TypeVar as _tp_TypeVar,
    Union as _tp_Union
)

# Libraries

from matplotlib.pyplot import (
    Axes as _mplp_Axes,
    Figure as _mplp_Figure
)

from networkx import (
    DiGraph as _nx_DiGraph,
    MultiDiGraph as _nx_MultiDiGraph
)

from numpy import (
    ndarray as _np_ndarray
)

from numpy.random import (
    RandomState as _npr_RandomState
)

from scipy.sparse import (
    spmatrix as _spsp_matrix
)

try:
    from pandas import (
        DataFrame as _pd_DataFrame,
        Series as _pd_Series
    )
    _pandas_found = True
except ImportError:  # noqa
    _pd_DataFrame = None
    _pd_Series = None
    _pandas_found = False

#########
# TYPES #
#########

# Generic

obool = _tp_Optional[bool]
ofloat = _tp_Optional[float]
oint = _tp_Optional[int]
ostr = _tp_Optional[str]

tscalar = _tp_Union[float, int]
oscalar = _tp_Optional[tscalar]

tany = _tp_Any
texception = Exception
titerable = _tp_Iterable

tarray = _np_ndarray
oarray = _tp_Optional[tarray]

tcache = _tp_Dict[str, tany]
ocache = _tp_Optional[tcache]

tfile = _tp_Tuple[str, str]
ofile = _tp_Optional[tfile]

tgraph = _nx_DiGraph
ograph = _tp_Optional[tgraph]

tgraphs = _tp_Union[tgraph, _nx_MultiDiGraph]
ographs = _tp_Optional[tgraphs]

tlimit_float = _tp_Tuple[float, bool]
olimit_float = _tp_Optional[tlimit_float]

tlimit_int = _tp_Tuple[int, bool]
olimit_int = _tp_Optional[tlimit_int]

tlimit_scalar = _tp_Tuple[tscalar, bool]
olimit_scalar = _tp_Optional[tlimit_scalar]

# noinspection PyTypeHints
tmc = _tp_TypeVar('MarkovChain')
omc = _tp_Optional[tmc]

tplot = _tp_Tuple[_mplp_Figure, _tp_Union[_mplp_Axes, _tp_List[_mplp_Axes]]]
oplot = _tp_Optional[tplot]

tnumeric = _tp_Union[_np_ndarray, _spsp_matrix] if not _pandas_found else _tp_Union[_np_ndarray, _spsp_matrix, _pd_DataFrame, _pd_Series]
onumeric = _tp_Optional[tnumeric]

trand = _npr_RandomState
orand = _tp_Optional[trand]

trandfunc = _tp_Callable
orandfunc = _tp_Optional[trandfunc]

trandfunc_flex = _tp_Union[_tp_Callable, str]
orandfunc_flex = _tp_Optional[trandfunc_flex]

# Lists

tlist_any = _tp_List[tany]
olist_any = _tp_Optional[tlist_any]

tlist_array = _tp_List[tarray]
olist_array = _tp_Optional[tlist_array]

tlist_float = _tp_List[float]
olist_float = _tp_Optional[tlist_float]

tlist_int = _tp_List[int]
olist_int = _tp_Optional[tlist_int]

tlist_mc = _tp_List[tmc]
olist_mc = _tp_Optional[tlist_mc]

tlist_str = _tp_List[str]
olist_str = _tp_Optional[tlist_str]

# Lists of Lists

tlists_any = _tp_List[tlist_any]
olists_any = _tp_Optional[tlists_any]

tlists_array = _tp_List[tlist_array]
olists_array = _tp_Optional[tlists_array]

tlists_float = _tp_List[tlist_float]
olists_float = _tp_Optional[tlists_float]

tlists_int = _tp_List[tlist_int]
olists_int = _tp_Optional[tlists_int]

tlists_str = _tp_List[tlist_str]
olists_str = _tp_Optional[tlists_str]

# Specific

tbcond = _tp_Union[float, int, str]
obcond = _tp_Optional[tbcond]

tdists_flex = _tp_Union[int, tlist_array]
odists_flex = _tp_Optional[tdists_flex]

tfitres = _tp_Tuple[oarray, ostr]
ofitres = _tp_Optional[tfitres]

tgenres = _tp_Tuple[oarray, ostr]
ogenres = _tp_Optional[tgenres]

tgenres_ext = _tp_Tuple[oarray, olist_str, ostr]
ogenres_ext = _tp_Optional[tgenres_ext]

tinterval = _tp_Tuple[tscalar, tscalar]
ointerval = _tp_Optional[tinterval]

tmc_dict = _tp_Dict[_tp_Tuple[str, str], float]
omc_dict = _tp_Optional[tmc_dict]

tmc_dict_flex = _tp_Dict[_tp_Tuple[str, str], tscalar]
omc_dict_flex = _tp_Optional[tmc_dict_flex]

tpart = _tp_List[_tp_Union[tlist_int, tlist_str]]
opart = _tp_Optional[tpart]

tparts = _tp_List[tpart]
oparts = _tp_Optional[tparts]

trdl = _tp_Tuple[tarray, tarray, tarray]
ordl = _tp_Optional[trdl]

tredists = _tp_Union[tarray, tlist_array]
oredists = _tp_Optional[tredists]

tstate = _tp_Union[int, str]
ostate = _tp_Optional[tstate]

tstates = _tp_Union[tstate, tlist_int, tlist_str]
ostates = _tp_Optional[tstates]

tstatus = _tp_Union[int, str, tnumeric]
ostatus = _tp_Optional[tstatus]

ttest = _tp_Tuple[obool, float, _tp_Dict[str, tany]]
otest = _tp_Optional[ttest]

ttest_chi2 = _tp_Tuple[float, float]
otest_chi2 = _tp_Optional[ttest_chi2]

ttfunc = _tp_Callable[[int, float, int, float], float]
otfunc = _tp_Optional[ttfunc]

ttimes_in = _tp_Union[int, tlist_int]
otimes_in = _tp_Optional[ttimes_in]

ttimes_out = _tp_Union[float, tlist_float]
otimes_out = _tp_Optional[ttimes_out]

tvalid_states = _tp_Tuple[tlist_int, tlist_str]
ovalid_states = _tp_Optional[tvalid_states]

tvalid_walk = _tp_Tuple[tlist_int, tlist_str]
ovalid_walk = _tp_Optional[tvalid_walk]

tvalid_walks = _tp_Tuple[tlists_int, tlist_str]
ovalid_walks = _tp_Optional[tvalid_walks]

twalk = _tp_Union[tlist_int, tlist_str]
owalk = _tp_Optional[twalk]

twalk_flex = _tp_Union[int, twalk]
owalk_flex = _tp_Optional[twalk_flex]

twalks = _tp_List[twalk]
owalks = _tp_Optional[twalks]

tweights = _tp_Union[float, int, tnumeric]
oweights = _tp_Optional[tweights]
