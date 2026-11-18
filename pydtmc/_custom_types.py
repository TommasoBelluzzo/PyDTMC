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
    # Compound Types -
    'tbndval', 'obndval',
    'tbndvals', 'obndvals',
    'tcache', 'ocache',
    'tdtype', 'odtype',
    'tedge_attrs', 'oedge_attrs',
    'tfile', 'ofile',
    'tinterval', 'ointerval',
    'tlim_float', 'olim_float',
    'tlim_int', 'olim_int',
    'tlim_scalar', 'olim_scalar',
    'trfunc', 'orfunc',
    'trfunc_f', 'orfunc_f',
    # Compound Types - Specific
    'tass_cp', 'oass_cp',
    'tass_ord', 'oass_ord',
    'tass_stat', 'oass_stat',
    'tbcond', 'obcond',
    'tfit_r', 'ofit_r',
    'thmm_dec', 'ohmm_dec',
    'thmm_dict', 'ohmm_dict',
    'thmm_dict_f', 'ohmm_dict_f',
    'thmm_gen', 'ohmm_gen',
    'thmm_gen_ext', 'ohmm_gen_ext',
    'thmm_gen_full', 'ohmm_gen_full',
    'thmm_params', 'ohmm_params',
    'thmm_params_res', 'ohmm_params_res',
    'thmm_pred', 'ohmm_pred',
    'thmm_seq', 'ohmm_seq',
    'thmm_seq_f', 'ohmm_seq_f',
    'thmm_step', 'ohmm_step',
    'thmm_sym', 'ohmm_sym',
    'thmm_sym_x', 'ohmm_sym_x',
    'tmc_dict', 'omc_dict',
    'tmc_dict_f', 'omc_dict_f',
    'tmc_gen', 'omc_gen',
    'tmc_gen_ext', 'omc_gen_ext',
    'tmc_gen_full', 'omc_gen_full',
    'tfc_prof', 'ofc_prof',
    'tobj_dict', 'oobj_dict',
    'tobs', 'oobs',
    'tpart', 'opart',
    'tparts', 'oparts',
    'tpprob', 'opprob',
    'tredists', 'oredists',
    'tsequence', 'osequence',
    'tsequences', 'osequences',
    'tstate', 'ostate',
    'tstates', 'ostates',
    'tstatus', 'ostatus',
    'ttfunc', 'otfunc',
    'ttimes_in', 'otimes_in',
    'ttimes_out', 'otimes_out',
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
import numpy.typing as _npt

try:
    import pandas as _pd
    _pandas_found = True
except ImportError:  # pragma: no cover
    _pd = None
    _pandas_found = False


#########
# TYPES #
#########

# BASE TYPES

tany = _tp.Any

obool = _tp.Optional[bool]
ofloat = _tp.Optional[float]
oint = _tp.Optional[int]
ostr = _tp.Optional[str]

tscalar = _tp.Union[float, int, _np.integer, _np.floating]
oscalar = _tp.Optional[tscalar]

tarray = _np.ndarray
oarray = _tp.Optional[tarray]

if _pandas_found:
    tnumeric = _tp.Union[_tp.List[tscalar], _tp.List[_tp.List[tscalar]], _np.ndarray, _spsp.spmatrix, _pd.DataFrame, _pd.Series]
else:
    tnumeric = _tp.Union[_tp.List[tscalar], _tp.List[_tp.List[tscalar]], _np.ndarray, _spsp.spmatrix]

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

thmm = _tp.ForwardRef('HiddenMarkovModel')
ohmm = _tp.Optional[thmm]

tmc = _tp.ForwardRef('MarkovChain')
omc = _tp.Optional[tmc]

tmodel = _tp.Union[thmm, tmc]
omodel = _tp.Optional[tmodel]

# PAIRS

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

# LISTS

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

# LISTS OF LISTS

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











# COMPOUND TYPES - GENERIC

tbndval = _tp.Tuple[float, _tp.Tuple[float, float]]
obndval = _tp.Optional[tbndval]

tbndvals = _tp.List[_tp.Tuple[float, _tp.Tuple[float, float]]]
obndvals = _tp.Optional[tbndvals]

tcache = _tp.Dict[str, tany]
ocache = _tp.Optional[tcache]

tdtype = _npt.DTypeLike
odtype = _tp.Optional[tdtype]

tfile = _tp.Tuple[tpath, str]
ofile = _tp.Optional[tfile]

tinterval = _tp.Tuple[tscalar, tscalar]
ointerval = _tp.Optional[tinterval]

tlim_float = _tp.Tuple[float, bool]
olim_float = _tp.Optional[tlim_float]

tlim_int = _tp.Tuple[int, bool]
olim_int = _tp.Optional[tlim_int]

tlim_scalar = _tp.Tuple[tscalar, bool]
olim_scalar = _tp.Optional[tlim_scalar]

trfunc = _tp.Callable
orfunc = _tp.Optional[trfunc]

trfunc_f = _tp.Union[_tp.Callable, str]
orfunc_f = _tp.Optional[trfunc_f]

tedge_attrs = _tp.List[_tp.Tuple[str, _tp.Tuple[str, ...]]]
oedge_attrs = _tp.Optional[tedge_attrs]

# COMPOUND TYPES - SPECIFIC

tass_cp = _tp.Tuple[oint, float, float]
oass_cp = _tp.Optional[tass_cp]

tass_ord = _tp.Tuple[int, _tp.Dict[int, float]]
oass_ord = _tp.Optional[tass_ord]

tass_stat = _tp.Tuple[obool, float, _tp.Dict[str, tany]]
oass_stat = _tp.Optional[tass_stat]

tbcond = _tp.Union[float, int, str]
obcond = _tp.Optional[tbcond]

tfit_r = _tp.Tuple[oarray, ostr]
ofit_r = _tp.Optional[tfit_r]

thmm_dec = _tp.Tuple[float, tarray, tarray, tarray, tarray, oarray]
ohmm_dec = _tp.Optional[thmm_dec]

thmm_dict = _tp.Dict[_tp.Tuple[str, str, str], float]
ohmm_dict = _tp.Optional[thmm_dict]

thmm_dict_f = _tp.Dict[_tp.Tuple[str, str, str], tscalar]
ohmm_dict_f = _tp.Optional[thmm_dict_f]

thmm_gen = _tp.Tuple[oarray, oarray, ostr]
ohmm_gen = _tp.Optional[thmm_gen]

thmm_gen_ext = _tp.Tuple[oarray, oarray, olist_str, olist_str]
ohmm_gen_ext = _tp.Optional[thmm_gen_ext]

thmm_gen_full = _tp.Tuple[oarray, oarray, olist_str, olist_str, ostr]
ohmm_gen_full = _tp.Optional[thmm_gen_full]

thmm_params = _tp.Tuple[tarray, tarray]
ohmm_params = _tp.Optional[thmm_params]

thmm_params_res = _tp.Tuple[oarray, oarray, ostr]
ohmm_params_res = _tp.Optional[thmm_params_res]

thmm_pred = _tp.Tuple[float, _tp.Union[tlist_int, tlist_str]]
ohmm_pred = _tp.Optional[thmm_pred]

thmm_seq = _tp.Tuple[tlist_int, tlist_int]
ohmm_seq = _tp.Optional[thmm_seq]

thmm_seq_f = _tp.Union[_tp.Tuple[tlist_int, tlist_int], _tp.Tuple[tlist_str, tlist_str]]
ohmm_seq_f = _tp.Optional[thmm_seq_f]

thmm_step = _tp.Union[int, str, _tp.Tuple[int, int], _tp.Tuple[str, str]]
ohmm_step = _tp.Optional[thmm_step]

thmm_sym = _tp.Union[tlist_int, tlist_str]
ohmm_sym = _tp.Optional[thmm_sym]

thmm_sym_x = _tp.Union[tlist_int, tlist_str, tlists_int, tlists_str]
ohmm_sym_x = _tp.Optional[thmm_sym_x]

tmc_dict = _tp.Dict[_tp.Tuple[str, str], float]
omc_dict = _tp.Optional[tmc_dict]

tmc_dict_f = _tp.Dict[_tp.Tuple[str, str], tscalar]
omc_dict_f = _tp.Optional[tmc_dict_f]

tmc_gen = _tp.Tuple[oarray, ostr]
omc_gen = _tp.Optional[tmc_gen]

tmc_gen_ext = _tp.Tuple[oarray, olist_str]
omc_gen_ext = _tp.Optional[tmc_gen_ext]

tmc_gen_full = _tp.Tuple[oarray, olist_str, ostr]
omc_gen_full = _tp.Optional[tmc_gen_full]

tfc_prof = _tp.Union[
    _tp.Tuple[_tp.Dict[_tp.Tuple[int, ...], float], _tp.List[_tp.Dict[_tp.Tuple[int, ...], float]]],
    _tp.Tuple[_tp.Dict[_tp.Tuple[str, ...], float], _tp.List[_tp.Dict[_tp.Tuple[str, ...], float]]]
]
ofc_prof = _tp.Optional[tfc_prof]

tobj_dict = _tp.Union[thmm_dict, tmc_dict]
oobj_dict = _tp.Optional[tobj_dict]

tobs = _tp.Union[str, tarray]
oobs = _tp.Optional[tobs]

tpart = _tp.Union[tlists_int, tlists_str]
opart = _tp.Optional[tpart]

tparts = _tp.List[tpart]
oparts = _tp.Optional[tparts]

tpprob = _tp.Dict[str, float]
opprob = _tp.Optional[tpprob]

tredists = _tp.Union[tarray, tlist_array]
oredists = _tp.Optional[tredists]

tstate = _tp.Union[int, str]
ostate = _tp.Optional[tstate]

tstates = _tp.Union[tstate, tlist_int, tlist_str]
ostates = _tp.Optional[tstates]

tstatus = _tp.Union[int, _np.integer, str, tnumeric]
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

tweights = _tp.Union[tscalar, tnumeric]
oweights = _tp.Optional[tweights]
