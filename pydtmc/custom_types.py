# -*- coding: utf-8 -*-

__all__ = [
    # Base Types
    'tany',
    'obool', 'ofloat', 'oint', 'ostr',
    'tscalar', 'oscalar',
    'tarray', 'oarray',
    'tnumeric', 'onumeric',
    'texception', 'oexception',
    'tgraph', 'ograph',
    'tgraphs', 'ographs',
    'tpath', 'opath',
    'tplot', 'oplot',
    'trand', 'orand',
    'tstack', 'ostack',
    'thmm', 'ohmm',
    'tmc', 'omc',
    'tmodel', 'omodel',
    # Pairs
    'tpair_array', 'opair_array',
    'tpair_bool', 'opair_bool',
    'tpair_float', 'opair_float',
    'tpair_int', 'opair_int',
    'tpair_str', 'opair_str',
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
    # Compound Types - Generic
    'tcache', 'ocache',
    'tdtype', 'odtype',
    'tedge_attributes', 'oedge_attributes',
    'tfile', 'ofile',
    'tinterval', 'ointerval',
    'tlimit_float', 'olimit_float',
    'tlimit_int', 'olimit_int',
    'tlimit_scalar', 'olimit_scalar',
    'tpartition', 'opartition',
    'tpartitions', 'opartitions',
    'trandfunc', 'orandfunc',
    'trandfunc_flex', 'orandfunc_flex',
    'trdl', 'ordl',
    'ttest', 'otest',
    'ttest_chi2', 'otest_chi2',
    # Compound Types - Specific
    'tbcond', 'obcond',
    'tdists_flex', 'odists_flex',
    'tfitting_res', 'ofitting_res',
    'thmm_decoding', 'ohmm_decoding',
    'thmm_dict', 'ohmm_dict',
    'thmm_dict_flex', 'ohmm_dict_flex',
    'thmm_generation', 'ohmm_generation',
    'thmm_generation_ext', 'ohmm_generation_ext',
    'thmm_params', 'ohmm_params',
    'thmm_params_res', 'ohmm_params_res',
    'thmm_sequence', 'ohmm_sequence',
    'thmm_sequence_ext', 'ohmm_sequence_ext',
    'thmm_step', 'ohmm_step',
    'thmm_symbols', 'ohmm_symbols',
    'thmm_symbols_ext', 'ohmm_symbols_ext',
    'thmm_symbols_out', 'ohmm_symbols_out',
    'thmm_viterbi', 'ohmm_viterbi',
    'thmm_viterbi_ext', 'ohmm_viterbi_ext',
    'tmc_dict', 'omc_dict',
    'tmc_dict_flex', 'omc_dict_flex',
    'tmc_generation', 'omc_generation',
    'tmc_generation_ext', 'omc_generation_ext',
    'tobj_dict', 'oobj_dict',
    'tredists', 'oredists',
    'tsequence', 'osequence',
    'tsequence_flex', 'osequence_flex',
    'tsequences', 'osequences',
    'tstate', 'ostate',
    'tstates', 'ostates',
    'tstatus', 'ostatus',
    'ttfunc', 'otfunc',
    'ttimes_in', 'otimes_in',
    'ttimes_out', 'otimes_out',
    'tvalid_states', 'ovalid_states',
    'tweights', 'oweights'
]


###########
# IMPORTS #
###########

# Standard


from inspect import (
    FrameInfo as _ins_FrameInfo
)

from pathlib import (
    Path as _pl_Path
)

from typing import (
    Any as _tp_Any,
    Callable as _tp_Callable,
    Dict as _tp_Dict,
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
except ImportError:  # pragma: no cover
    _pd_DataFrame, _pd_Series = None, None
    _pandas_found = False


#########
# TYPES #
#########

# Base Types

tany = _tp_Any

obool = _tp_Optional[bool]
ofloat = _tp_Optional[float]
oint = _tp_Optional[int]
ostr = _tp_Optional[str]

tscalar = _tp_Union[float, int]
oscalar = _tp_Optional[tscalar]

tarray = _np_ndarray
oarray = _tp_Optional[tarray]

tnumeric = _tp_Union[_np_ndarray, _spsp_matrix] if not _pandas_found else _tp_Union[_np_ndarray, _spsp_matrix, _pd_DataFrame, _pd_Series]
onumeric = _tp_Optional[tnumeric]

texception = Exception
oexception = _tp_Optional[texception]

tgraph = _nx_DiGraph
ograph = _tp_Optional[tgraph]

tgraphs = _tp_Union[tgraph, _nx_MultiDiGraph]
ographs = _tp_Optional[tgraphs]

tpath = _tp_Union[str, _pl_Path]
opath = _tp_Optional[tpath]

tplot = _tp_Tuple[_mplp_Figure, _tp_Union[_mplp_Axes, _tp_List[_mplp_Axes]]]
oplot = _tp_Optional[tplot]

trand = _npr_RandomState
orand = _tp_Optional[trand]

tstack = _tp_List[_ins_FrameInfo]
ostack = _tp_Optional[tstack]

# noinspection PyTypeHints
thmm = _tp_TypeVar('HiddenMarkovModel')
ohmm = _tp_Optional[thmm]

# noinspection PyTypeHints
tmc = _tp_TypeVar('MarkovChain')
omc = _tp_Optional[tmc]

tmodel = _tp_Union[tmc, thmm]
omodel = _tp_Optional[tmodel]

# Pairs

tpair_array = _tp_Tuple[tarray, tarray]
opair_array = _tp_Optional[tpair_array]

tpair_bool = _tp_Tuple[bool, bool]
opair_bool = _tp_Optional[tpair_bool]

tpair_float = _tp_Tuple[float, float]
opair_float = _tp_Optional[tpair_float]

tpair_int = _tp_Tuple[int, int]
opair_int = _tp_Optional[tpair_int]

tpair_str = _tp_Tuple[str, str]
opair_str = _tp_Optional[tpair_str]

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

# Compound Types - Generic

tcache = _tp_Dict[str, tany]
ocache = _tp_Optional[tcache]

tdtype = _tp_Union[object, str]
odtype = _tp_Optional[tdtype]

tedge_attributes = _tp_List[_tp_Tuple[str, _tp_Tuple[str, ...]]]
oedge_attributes = _tp_Optional[tedge_attributes]

tfile = _tp_Tuple[tpath, str]
ofile = _tp_Optional[tfile]

tinterval = _tp_Tuple[tscalar, tscalar]
ointerval = _tp_Optional[tinterval]

tlimit_float = _tp_Tuple[float, bool]
olimit_float = _tp_Optional[tlimit_float]

tlimit_int = _tp_Tuple[int, bool]
olimit_int = _tp_Optional[tlimit_int]

tlimit_scalar = _tp_Tuple[tscalar, bool]
olimit_scalar = _tp_Optional[tlimit_scalar]

tpartition = _tp_Union[tlists_int, tlists_str]
opartition = _tp_Optional[tpartition]

tpartitions = _tp_List[tpartition]
opartitions = _tp_Optional[tpartitions]

trandfunc = _tp_Callable
orandfunc = _tp_Optional[trandfunc]

trandfunc_flex = _tp_Union[_tp_Callable, str]
orandfunc_flex = _tp_Optional[trandfunc_flex]

trdl = _tp_Tuple[tarray, tarray, tarray]
ordl = _tp_Optional[trdl]

ttest = _tp_Tuple[obool, float, _tp_Dict[str, tany]]
otest = _tp_Optional[ttest]

ttest_chi2 = _tp_Tuple[float, float]
otest_chi2 = _tp_Optional[ttest_chi2]

# Compound Types - Specific

tbcond = _tp_Union[float, int, str]
obcond = _tp_Optional[tbcond]

tdists_flex = _tp_Union[int, tlist_array]
odists_flex = _tp_Optional[tdists_flex]

tfitting_res = _tp_Tuple[oarray, ostr]
ofitting_res = _tp_Optional[tfitting_res]

thmm_decoding = _tp_Union[_tp_Tuple[float, tarray, tarray, tarray, tarray], _tp_Tuple[float, tarray, tarray, tarray]]
ohmm_decoding = _tp_Optional[thmm_decoding]

thmm_dict = _tp_Dict[_tp_Tuple[str, str, str], float]
ohmm_dict = _tp_Optional[thmm_dict]

thmm_dict_flex = _tp_Dict[_tp_Tuple[str, str, str], tscalar]
ohmm_dict_flex = _tp_Optional[thmm_dict_flex]

thmm_generation = _tp_Tuple[tarray, tarray, tlist_str, tlist_str]
ohmm_generation = _tp_Optional[thmm_generation]

thmm_generation_ext = _tp_Tuple[oarray, oarray, olist_str, olist_str, ostr]
ohmm_generation_ext = _tp_Optional[thmm_generation_ext]

thmm_params = _tp_Tuple[tarray, tarray]
ohmm_params = _tp_Optional[thmm_params]

thmm_params_res = _tp_Tuple[oarray, oarray, ostr]
ohmm_params_res = _tp_Optional[thmm_params_res]

thmm_sequence = _tp_Tuple[tlist_int, tlist_int]
ohmm_sequence = _tp_Optional[thmm_sequence]

thmm_sequence_ext = _tp_Union[_tp_Tuple[tlist_int, tlist_int], _tp_Tuple[tlist_str, tlist_str]]
ohmm_sequence_ext = _tp_Optional[thmm_sequence_ext]

thmm_step = _tp_Union[int, str, _tp_Tuple[int, int], _tp_Tuple[str, str]]
ohmm_step = _tp_Optional[thmm_step]

thmm_symbols = _tp_Union[tlist_int, tlist_str]
ohmm_symbols = _tp_Optional[thmm_symbols]

thmm_symbols_ext = _tp_Union[tlist_int, tlist_str, tlists_int, tlists_str]
ohmm_symbols_ext = _tp_Optional[thmm_symbols_ext]

thmm_symbols_out = _tp_Union[tlist_int, tlists_int]
ohmm_symbols_out = _tp_Optional[thmm_symbols_out]

thmm_viterbi = _tp_Tuple[float, tlist_int]
ohmm_viterbi = _tp_Optional[thmm_viterbi]

thmm_viterbi_ext = _tp_Tuple[float, _tp_Union[tlist_int, tlist_str]]
ohmm_viterbi_ext = _tp_Optional[thmm_viterbi_ext]

tmc_dict = _tp_Dict[_tp_Tuple[str, str], float]
omc_dict = _tp_Optional[tmc_dict]

tmc_dict_flex = _tp_Dict[_tp_Tuple[str, str], tscalar]
omc_dict_flex = _tp_Optional[tmc_dict_flex]

tmc_generation = _tp_Tuple[oarray, ostr]
omc_generation = _tp_Optional[tmc_generation]

tmc_generation_ext = _tp_Tuple[oarray, olist_str, ostr]
omc_generation_ext = _tp_Optional[tmc_generation_ext]

tobj_dict = _tp_Union[thmm_dict, tmc_dict]
oobj_dict = _tp_Optional[tobj_dict]

tredists = _tp_Union[tarray, tlist_array]
oredists = _tp_Optional[tredists]

tstate = _tp_Union[int, str]
ostate = _tp_Optional[tstate]

tstates = _tp_Union[tstate, tlist_int, tlist_str]
ostates = _tp_Optional[tstates]

tstatus = _tp_Union[int, str, tnumeric]
ostatus = _tp_Optional[tstatus]

ttfunc = _tp_Callable[[int, float, int, float], float]
otfunc = _tp_Optional[ttfunc]

tsequence = _tp_Union[tlist_int, tlist_str]
osequence = _tp_Optional[tsequence]

tsequence_flex = _tp_Union[int, tsequence]
osequence_flex = _tp_Optional[tsequence_flex]

tsequences = _tp_List[tsequence]
osequences = _tp_Optional[tsequences]

ttimes_in = _tp_Union[int, tlist_int]
otimes_in = _tp_Optional[ttimes_in]

ttimes_out = _tp_Union[float, tlist_float]
otimes_out = _tp_Optional[ttimes_out]

tvalid_states = _tp_Tuple[tlist_int, tlist_str]
ovalid_states = _tp_Optional[tvalid_states]

tweights = _tp_Union[float, int, tnumeric]
oweights = _tp_Optional[tweights]
