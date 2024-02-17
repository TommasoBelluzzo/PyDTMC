# -*- coding: utf-8 -*-

from __future__ import annotations

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
    'tlist_str', 'olist_str',
    'tlist_hmm', 'olist_hmm',
    'tlist_mc', 'olist_mc',
    'tlist_model', 'olist_model',
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
    'tfitting_res', 'ofitting_res',
    'thmm_decoding', 'ohmm_decoding',
    'thmm_dict', 'ohmm_dict',
    'thmm_dict_flex', 'ohmm_dict_flex',
    'thmm_generation', 'ohmm_generation',
    'thmm_params', 'ohmm_params',
    'thmm_params_res', 'ohmm_params_res',
    'thmm_prediction', 'ohmm_prediction',
    'thmm_sequence', 'ohmm_sequence',
    'thmm_sequence_ext', 'ohmm_sequence_ext',
    'thmm_step', 'ohmm_step',
    'thmm_symbols', 'ohmm_symbols',
    'thmm_symbols_ext', 'ohmm_symbols_ext',
    'tmc_dict', 'omc_dict',
    'tmc_dict_flex', 'omc_dict_flex',
    'tmc_generation', 'omc_generation',
    'tobj_dict', 'oobj_dict',
    'tredists', 'oredists',
    'tsequence', 'osequence',
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

import inspect as _ins
import pathlib as _pl
import typing as _tp

# Libraries

import matplotlib.pyplot as _mplp
import networkx as _nx
import numpy as _np
import numpy.random as _npr
import scipy.sparse as _spsp

try:
    import pandas as _pd
    _pandas_found = True
except ImportError:  # pragma: no cover
    _pd = None
    _pandas_found = False


#########
# TYPES #
#########

# Base Types

tany = _tp.Any

obool = _tp.Optional[bool]
ofloat = _tp.Optional[float]
oint = _tp.Optional[int]
ostr = _tp.Optional[str]

tscalar = _tp.Union[float, int]
oscalar = _tp.Optional[tscalar]

tarray = _np.ndarray
oarray = _tp.Optional[tarray]

tnumeric = _tp.Union[_np.ndarray, _spsp.spmatrix] if not _pandas_found else _tp.Union[_np.ndarray, _spsp.spmatrix, _pd.DataFrame, _pd.Series]
onumeric = _tp.Optional[tnumeric]

texception = Exception
oexception = _tp.Optional[texception]

tgraph = _nx.DiGraph
ograph = _tp.Optional[tgraph]

tgraphs = _tp.Union[tgraph, _nx.MultiDiGraph]
ographs = _tp.Optional[tgraphs]

tpath = _tp.Union[str, _pl.Path]
opath = _tp.Optional[tpath]

tplot = _tp.Tuple[_mplp.Figure, _tp.Union[_mplp.Axes, _tp.List[_mplp.Axes]]]
oplot = _tp.Optional[tplot]

trand = _npr.RandomState  # pylint: disable=no-member
orand = _tp.Optional[trand]

tstack = _tp.List[_ins.FrameInfo]
ostack = _tp.Optional[tstack]

# noinspection PyTypeHints
thmm = _tp.ForwardRef('HiddenMarkovModel')
ohmm = _tp.Optional[thmm]

# noinspection PyTypeHints
tmc = _tp.ForwardRef('MarkovChain')
omc = _tp.Optional[tmc]

tmodel = _tp.Union[thmm, tmc]
omodel = _tp.Optional[tmodel]

# Pairs

tpair_array = _tp.Tuple[tarray, tarray]
opair_array = _tp.Optional[tpair_array]

tpair_bool = _tp.Tuple[bool, bool]
opair_bool = _tp.Optional[tpair_bool]

tpair_float = _tp.Tuple[float, float]
opair_float = _tp.Optional[tpair_float]

tpair_int = _tp.Tuple[int, int]
opair_int = _tp.Optional[tpair_int]

tpair_str = _tp.Tuple[str, str]
opair_str = _tp.Optional[tpair_str]

# Lists

tlist_any = _tp.List[tany]
olist_any = _tp.Optional[tlist_any]

tlist_array = _tp.List[tarray]
olist_array = _tp.Optional[tlist_array]

tlist_float = _tp.List[float]
olist_float = _tp.Optional[tlist_float]

tlist_int = _tp.List[int]
olist_int = _tp.Optional[tlist_int]

tlist_str = _tp.List[str]
olist_str = _tp.Optional[tlist_str]

tlist_hmm = _tp.List[thmm]
olist_hmm = _tp.Optional[tlist_hmm]

tlist_mc = _tp.List[tmc]
olist_mc = _tp.Optional[tlist_mc]

tlist_model = _tp.List[tmodel]
olist_model = _tp.Optional[tlist_model]

# Lists of Lists

tlists_any = _tp.List[tlist_any]
olists_any = _tp.Optional[tlists_any]

tlists_array = _tp.List[tlist_array]
olists_array = _tp.Optional[tlists_array]

tlists_float = _tp.List[tlist_float]
olists_float = _tp.Optional[tlists_float]

tlists_int = _tp.List[tlist_int]
olists_int = _tp.Optional[tlists_int]

tlists_str = _tp.List[tlist_str]
olists_str = _tp.Optional[tlists_str]

# Compound Types - Generic

tcache = _tp.Dict[str, tany]
ocache = _tp.Optional[tcache]

tdtype = _tp.Union[object, str]
odtype = _tp.Optional[tdtype]

tedge_attributes = _tp.List[_tp.Tuple[str, _tp.Tuple[str, ...]]]
oedge_attributes = _tp.Optional[tedge_attributes]

tfile = _tp.Tuple[tpath, str]
ofile = _tp.Optional[tfile]

tinterval = _tp.Tuple[tscalar, tscalar]
ointerval = _tp.Optional[tinterval]

tlimit_float = _tp.Tuple[float, bool]
olimit_float = _tp.Optional[tlimit_float]

tlimit_int = _tp.Tuple[int, bool]
olimit_int = _tp.Optional[tlimit_int]

tlimit_scalar = _tp.Tuple[tscalar, bool]
olimit_scalar = _tp.Optional[tlimit_scalar]

tpartition = _tp.Union[tlists_int, tlists_str]
opartition = _tp.Optional[tpartition]

tpartitions = _tp.List[tpartition]
opartitions = _tp.Optional[tpartitions]

trandfunc = _tp.Callable
orandfunc = _tp.Optional[trandfunc]

trandfunc_flex = _tp.Union[_tp.Callable, str]
orandfunc_flex = _tp.Optional[trandfunc_flex]

trdl = _tp.Tuple[tarray, tarray, tarray]
ordl = _tp.Optional[trdl]

ttest = _tp.Tuple[obool, float, _tp.Dict[str, tany]]
otest = _tp.Optional[ttest]

ttest_chi2 = _tp.Tuple[float, float]
otest_chi2 = _tp.Optional[ttest_chi2]

# Compound Types - Specific

tbcond = _tp.Union[float, int, str]
obcond = _tp.Optional[tbcond]

tfitting_res = _tp.Tuple[oarray, ostr]
ofitting_res = _tp.Optional[tfitting_res]

thmm_decoding = _tp.Tuple[float, tarray, tarray, tarray, oarray]
ohmm_decoding = _tp.Optional[thmm_decoding]

thmm_dict = _tp.Dict[_tp.Tuple[str, str, str], float]
ohmm_dict = _tp.Optional[thmm_dict]

thmm_dict_flex = _tp.Dict[_tp.Tuple[str, str, str], tscalar]
ohmm_dict_flex = _tp.Optional[thmm_dict_flex]

thmm_generation = _tp.Tuple[oarray, oarray, olist_str, olist_str, ostr]
ohmm_generation = _tp.Optional[thmm_generation]

thmm_params = _tp.Tuple[tarray, tarray]
ohmm_params = _tp.Optional[thmm_params]

thmm_params_res = _tp.Tuple[oarray, oarray, ostr]
ohmm_params_res = _tp.Optional[thmm_params_res]

thmm_prediction = _tp.Tuple[float, _tp.Union[tlist_int, tlist_str]]
ohmm_prediction = _tp.Optional[thmm_prediction]

thmm_sequence = _tp.Tuple[tlist_int, tlist_int]
ohmm_sequence = _tp.Optional[thmm_sequence]

thmm_sequence_ext = _tp.Union[_tp.Tuple[tlist_int, tlist_int], _tp.Tuple[tlist_str, tlist_str]]
ohmm_sequence_ext = _tp.Optional[thmm_sequence_ext]

thmm_step = _tp.Union[int, str, _tp.Tuple[int, int], _tp.Tuple[str, str]]
ohmm_step = _tp.Optional[thmm_step]

thmm_symbols = _tp.Union[tlist_int, tlist_str]
ohmm_symbols = _tp.Optional[thmm_symbols]

thmm_symbols_ext = _tp.Union[tlist_int, tlist_str, tlists_int, tlists_str]
ohmm_symbols_ext = _tp.Optional[thmm_symbols_ext]

tmc_dict = _tp.Dict[_tp.Tuple[str, str], float]
omc_dict = _tp.Optional[tmc_dict]

tmc_dict_flex = _tp.Dict[_tp.Tuple[str, str], tscalar]
omc_dict_flex = _tp.Optional[tmc_dict_flex]

tmc_generation = _tp.Tuple[oarray, olist_str, ostr]
omc_generation = _tp.Optional[tmc_generation]

tobj_dict = _tp.Union[thmm_dict, tmc_dict]
oobj_dict = _tp.Optional[tobj_dict]

tredists = _tp.Union[tarray, tlist_array]
oredists = _tp.Optional[tredists]

tstate = _tp.Union[int, str]
ostate = _tp.Optional[tstate]

tstates = _tp.Union[tstate, tlist_int, tlist_str]
ostates = _tp.Optional[tstates]

tstatus = _tp.Union[int, str, tnumeric]
ostatus = _tp.Optional[tstatus]

ttfunc = _tp.Callable[[int, float, int, float], float]
otfunc = _tp.Optional[ttfunc]

tsequence = _tp.Union[tlist_int, tlist_str]
osequence = _tp.Optional[tsequence]

tsequences = _tp.List[tsequence]
osequences = _tp.Optional[tsequences]

ttimes_in = _tp.Union[int, tlist_int]
otimes_in = _tp.Optional[ttimes_in]

ttimes_out = _tp.Union[float, tlist_float]
otimes_out = _tp.Optional[ttimes_out]

tvalid_states = _tp.Tuple[tlist_int, tlist_str]
ovalid_states = _tp.Optional[tvalid_states]

tweights = _tp.Union[float, int, tnumeric]
oweights = _tp.Optional[tweights]
