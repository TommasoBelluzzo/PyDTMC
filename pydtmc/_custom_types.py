# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    # Base Types
    'tany',
    'obool', 'ofloat', 'oint', 'ostr',
    'tdtype', 'odtype',
    'texc', 'oexc',
    'tgraph', 'ograph',
    'tgraphs', 'ographs',
    'tpath', 'opath',
    'tplot', 'oplot',
    'trand', 'orand',
    'tstack', 'ostack',
    # Numeric Types
    'tarray', 'oarray',
    'tnum', 'onum',
    'tvec', 'ovec',
    'tmat', 'omat',
    # Library Types
    'thmm', 'ohmm',
    'tmc', 'omc',
    'tmdl', 'omdl',
    # Lists
    'tlist_any', 'olist_any',
    'tlist_array', 'olist_array',
    'tlist_float', 'olist_float',
    'tlist_int', 'olist_int',
    'tlist_str', 'olist_str',
    'tlist_hmm', 'olist_hmm',
    'tlist_mc', 'olist_mc',
    'tlist_mdl', 'olist_mdl',
    # Lists of Lists
    'tlists_any', 'olists_any',
    'tlists_array', 'olists_array',
    'tlists_float', 'olists_float',
    'tlists_int', 'olists_int',
    'tlists_str', 'olists_str',
    # Pairs
    'tpair_array', 'opair_array',
    'tpair_bool', 'opair_bool',
    'tpair_float', 'opair_float',
    'tpair_int', 'opair_int',
    'tpair_str', 'opair_str',
    # Tuples
    'ttup_array', 'otup_array',
    'ttup_bool', 'otup_bool',
    'ttup_float', 'otup_float',
    'ttup_int', 'otup_int',
    'ttup_str', 'otup_str',
    # Compound Types - Generic
    'tbndval', 'obndval',
    'tbndvals', 'obndvals',
    'tcache', 'ocache',
    'tedge_attrs', 'oedge_attrs',
    'tfile', 'ofile',
    'tival', 'oival',
    'tlim_float', 'olim_float',
    'tlim_int', 'olim_int',
    'tlim_num', 'olim_num',
    'trfunc', 'orfunc',
    'trfunc_f', 'orfunc_f',
    # Compound Types - Specific
    'tass_cp', 'oass_cp',
    'tass_ord', 'oass_ord',
    'tass_stat', 'oass_stat',
    'tbcond', 'obcond',
    'tfit_r', 'ofit_r',
    'thmm_dec', 'ohmm_dec',
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
    'tmc_gen', 'omc_gen',
    'tmc_gen_ext', 'omc_gen_ext',
    'tmc_gen_full', 'omc_gen_full',
    'tobs', 'oobs',
    'tpart', 'opart',
    'tparts', 'oparts',
    'tpprobs', 'opprobs',
    'tredists', 'oredists',
    'tseq', 'oseq',
    'tseqs', 'oseqs',
    'tstate', 'ostate',
    'tstates', 'ostates',
    'tstatus', 'ostatus',
    'ttfunc', 'otfunc',
    'ttimes_in', 'otimes_in',
    'ttimes_out', 'otimes_out',
    'twgts', 'owgts',
    # Compound Types - Nested
    'tmdl_dict', 'omdl_dict',
    'tmdl_dict_f', 'omdl_dict_f',
    'thmm_dict', 'ohmm_dict',
    'thmm_dict_f', 'ohmm_dict_f',
    'tmc_dict', 'omc_dict',
    'tmc_dict_f', 'omc_dict_f',
    'tfc_prof', 'ofc_prof'
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
import numpy.typing as _npt
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

# BASE TYPES

tany = _tp.Any

obool = _tp.Optional[bool]
ofloat = _tp.Optional[float]
oint = _tp.Optional[int]
ostr = _tp.Optional[str]

tdtype = _npt.DTypeLike
odtype = _tp.Optional[tdtype]

texc = Exception
oexc = _tp.Optional[texc]

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

# NUMERIC TYPES

tarray = _np.ndarray
oarray = _tp.Optional[tarray]

tnum = _tp.Union[float, int]
onum = _tp.Optional[tnum]

if _pandas_found:
    tvec = _tp.Union[_tp.List[tnum], tarray, _pd.Series]
    tmat = _tp.Union[_tp.List[_tp.List[tnum]], tarray, _spsp.spmatrix, _pd.DataFrame]
else:
    tvec = _tp.Union[_tp.List[tnum], tarray]
    tmat = _tp.Union[_tp.List[_tp.List[tnum]], tarray, _spsp.spmatrix]

ovec = _tp.Optional[tvec]
omat = _tp.Optional[tmat]

# LIBRARY TYPES

if _tp.TYPE_CHECKING:

    from .hidden_markov_model import HiddenMarkovModel
    from .markov_chain import MarkovChain

    thmm: _tp.TypeAlias = HiddenMarkovModel
    tmc: _tp.TypeAlias = MarkovChain

else:
    thmm = _tp.ForwardRef('HiddenMarkovModel')
    tmc = _tp.ForwardRef('MarkovChain')

ohmm = _tp.Optional[thmm]
omc = _tp.Optional[tmc]

tmdl = _tp.Union[thmm, tmc]
omdl = _tp.Optional[tmdl]

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

tlist_mdl = _tp.List[tmdl]
olist_mdl = _tp.Optional[tlist_mdl]

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

# TUPLES

ttup_array = _tp.Tuple[tarray, ...]
otup_array = _tp.Optional[ttup_array]

ttup_bool = _tp.Tuple[bool, ...]
otup_bool = _tp.Optional[ttup_bool]

ttup_float = _tp.Tuple[float, ...]
otup_float = _tp.Optional[ttup_float]

ttup_int = _tp.Tuple[int, ...]
otup_int = _tp.Optional[ttup_int]

ttup_str = _tp.Tuple[str, ...]
otup_str = _tp.Optional[ttup_str]

# COMPOUND TYPES - GENERIC

tbndval = _tp.Tuple[float, tpair_float]
obndval = _tp.Optional[tbndval]

tbndvals = _tp.List[tbndval]
obndvals = _tp.Optional[tbndvals]

tcache = _tp.Dict[str, tany]
ocache = _tp.Optional[tcache]

tedge_attrs = _tp.List[_tp.Tuple[str, ttup_str]]
oedge_attrs = _tp.Optional[tedge_attrs]

tfile = _tp.Tuple[_pl.Path, str]
ofile = _tp.Optional[tfile]

tival = _tp.Tuple[tnum, tnum]
oival = _tp.Optional[tival]

tlim_float = _tp.Tuple[float, bool]
olim_float = _tp.Optional[tlim_float]

tlim_int = _tp.Tuple[int, bool]
olim_int = _tp.Optional[tlim_int]

tlim_num = _tp.Tuple[tnum, bool]
olim_num = _tp.Optional[tlim_num]

trfunc = _tp.Callable
orfunc = _tp.Optional[trfunc]

trfunc_f = _tp.Union[trfunc, str]
orfunc_f = _tp.Optional[trfunc_f]

# COMPOUND TYPES - SPECIFIC

tass_cp = _tp.Tuple[oint, float, float]
oass_cp = _tp.Optional[tass_cp]

tass_ord = _tp.Tuple[int, _tp.Dict[int, float]]
oass_ord = _tp.Optional[tass_ord]

tass_stat = _tp.Tuple[obool, float, _tp.Dict[str, tany]]
oass_stat = _tp.Optional[tass_stat]

tbcond = _tp.Union[tnum, str]
obcond = _tp.Optional[tbcond]

tfit_r = _tp.Tuple[oarray, ostr]
ofit_r = _tp.Optional[tfit_r]

thmm_dec = _tp.Tuple[float, tarray, tarray, tarray, tarray, oarray]
ohmm_dec = _tp.Optional[thmm_dec]

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

thmm_seq_f = _tp.Union[thmm_seq, _tp.Tuple[tlist_str, tlist_str]]
ohmm_seq_f = _tp.Optional[thmm_seq_f]

thmm_step = _tp.Union[int, str, tpair_int, tpair_str]
ohmm_step = _tp.Optional[thmm_step]

thmm_sym = _tp.Union[tlist_int, tlist_str]
ohmm_sym = _tp.Optional[thmm_sym]

thmm_sym_x = _tp.Union[thmm_sym, tlists_int, tlists_str]
ohmm_sym_x = _tp.Optional[thmm_sym_x]

tmc_gen = _tp.Tuple[oarray, ostr]
omc_gen = _tp.Optional[tmc_gen]

tmc_gen_ext = _tp.Tuple[oarray, olist_str]
omc_gen_ext = _tp.Optional[tmc_gen_ext]

tmc_gen_full = _tp.Tuple[oarray, olist_str, ostr]
omc_gen_full = _tp.Optional[tmc_gen_full]

tobs = _tp.Union[str, tmat]
oobs = _tp.Optional[tobs]

tpart = _tp.Union[tlists_int, tlists_str]
opart = _tp.Optional[tpart]

tparts = _tp.List[tpart]
oparts = _tp.Optional[tparts]

tpprobs = _tp.Dict[str, float]
opprobs = _tp.Optional[tpprobs]

tredists = _tp.Union[tarray, tlist_array]
oredists = _tp.Optional[tredists]

tseq = _tp.Union[tlist_int, tlist_str]
oseq = _tp.Optional[tseq]

tseqs = _tp.List[tseq]
oseqs = _tp.Optional[tseqs]

tstate = _tp.Union[int, str]
ostate = _tp.Optional[tstate]

tstates = _tp.Union[tstate, tlist_int, tlist_str]
ostates = _tp.Optional[tstates]

tstatus = _tp.Union[tstate, tvec]
ostatus = _tp.Optional[tstatus]

ttfunc = _tp.Callable[[int, float, int, float], tnum]
otfunc = _tp.Optional[ttfunc]

ttimes_in = _tp.Union[int, tlist_int]
otimes_in = _tp.Optional[ttimes_in]

ttimes_out = _tp.Union[float, tlist_float]
otimes_out = _tp.Optional[ttimes_out]

twgts = _tp.Union[tnum, tvec]
owgts = _tp.Optional[twgts]

# COMPOUND TYPES - NESTED

_mdk = _tp.TypeVar('_mdk', tpair_str, _tp.Tuple[str, str, str])

tmdl_dict = _tp.Dict[_mdk, float]
omdl_dict = _tp.Optional[tmdl_dict]
tmdl_dict_f = _tp.Dict[_mdk, tnum]
omdl_dict_f = _tp.Optional[tmdl_dict_f]

thmm_dict = tmdl_dict[_tp.Tuple[str, str, str]]
ohmm_dict = _tp.Optional[thmm_dict]
thmm_dict_f = tmdl_dict_f[_tp.Tuple[str, str, str]]
ohmm_dict_f = _tp.Optional[thmm_dict_f]

tmc_dict = tmdl_dict[tpair_str]
omc_dict = _tp.Optional[tmc_dict]
tmc_dict_f = tmdl_dict_f[tpair_str]
omc_dict_f = _tp.Optional[tmc_dict_f]

_fck = _tp.TypeVar('_fck', ttup_int, ttup_str)
_fc_prof = _tp.Tuple[_tp.Dict[_fck, float], _tp.List[_tp.Dict[_fck, float]]]

tfc_prof = _tp.Union[_fc_prof[ttup_int], _fc_prof[ttup_str]]
ofc_prof = _tp.Optional[tfc_prof]
