# -*- coding: utf-8 -*-

__all__ = [
    'build_hmm_graph',
    'build_mc_graph',
    'create_labels',
    'create_labels_from_data',
    'create_models_names',
    'create_rng',
    'create_validation_error',
    'get_caller',
    'get_file_extension',
    'get_full_name',
    'get_instance_generators',
    'get_numpy_random_distributions',
    'extract_numeric',
    'is_array',
    'is_boolean',
    'is_dictionary',
    'is_float',
    'is_graph',
    'is_integer',
    'is_list',
    'is_number',
    'is_pandas',
    'is_spmatrix',
    'is_string',
    'is_tuple'
]


###########
# IMPORTS #
###########

# Standard

import inspect as _ins
import pathlib as _pl

# Libraries

import networkx as _nx
import numpy as _np
import numpy.random as _npr
import numpy.random.mtrand as _nprm
import scipy.sparse as _spsp

try:
    import pandas as _pd
    _pandas_found = True
except ImportError:  # pragma: no cover
    _pd = None
    _pandas_found = False

# Internal

from .custom_types import (
    odtype as _odtype,
    oint as _oint,
    olist_str as _olist_str,
    tany as _tany,
    tarray as _tarray,
    texception as _texception,
    tgraph as _tgraph,
    tlist_model as _tlist_model,
    tlist_str as _tlist_str,
    tpair_bool as _tpair_bool,
    tpath as _tpath,
    trand as _trand,
    tstack as _tstack
)

from .exceptions import (
    ValidationError as _ValidationError
)


#############
# FUNCTIONS #
#############

def build_hmm_graph(p, e, states, symbols):

    n, k = len(states), len(symbols)

    graph = _nx.DiGraph()
    graph.add_nodes_from(states, layer=1)
    graph.add_nodes_from(symbols, layer=0)

    for i in range(n):

        state_i = states[i]

        for j in range(n):

            p_ij = p[i, j]

            if p_ij > 0.0:
                graph.add_edge(state_i, states[j], type='P', weight=p_ij)

        for j in range(k):

            e_ij = e[i, j]

            if e_ij > 0.0:
                graph.add_edge(state_i, symbols[j], type='E', weight=e_ij)

    return graph


def build_mc_graph(p: _tarray, states: _tlist_str) -> _tgraph:

    n = len(states)

    graph = _nx.DiGraph()
    graph.add_nodes_from(states)

    for i in range(n):

        state_i = states[i]

        for j in range(n):

            p_ij = p[i, j]

            if p_ij > 0.0:
                graph.add_edge(state_i, states[j], weight=p_ij)

    return graph


def create_labels(count: int, prefix: str = '') -> _tlist_str:

    labels = [f'{prefix}{i:d}' for i in range(1, count + 1)]

    return labels


def create_labels_from_data(data: _tany, prefix: str = '') -> _olist_str:

    if not is_list(data):
        return None

    if all(is_integer(state) for state in data):
        labels = [f'{prefix}{i:d}' for i in range(1, len(set(data)) + 1)]
    elif all(is_string(state) for state in data):
        labels = [f'{prefix}_{item}' if len(prefix) > 0 else f'{item}' for item in sorted(set(data))]
    else:
        return None

    labels_length = len(labels)

    if labels_length < 2:
        return None

    labels_unique_length = len(set(data))

    if labels_unique_length < labels_length:
        return None

    return labels


def create_models_names(models: _tlist_model) -> _tlist_str:

    if all(model.__class__.__name__ == 'HiddenMarkovModel' for model in models):
        models_label = 'HMM'
    elif all(model.__class__.__name__ == 'MarkovChain' for model in models):
        models_label = 'MC'
    else:
        models_label = 'MODEL'

    models_names = [f'$\mathregular{{{models_label}_{index + 1}}}$' for index, model in enumerate(models)]

    return models_names


# noinspection PyProtectedMember
def create_rng(seed: _oint) -> _trand:

    if seed is None:
        rng = _nprm._rand  # pylint: disable=c-extension-no-member
    elif is_integer(seed):
        rng = _npr.RandomState(int(seed))  # pylint: disable=no-member
    else:
        raise TypeError('The specified seed is not a valid RNG initializer.')

    return rng


def create_validation_error(ex: _texception, trace: _tany) -> _ValidationError:

    arguments = ''.join(trace[0][4]).split('=', 1)[0].strip()

    if ',' in arguments:
        arguments = arguments[:arguments.index(',')]

    message = str(ex).replace('@arg@', arguments)
    validation_error = _ValidationError(message)

    return validation_error


def extract_numeric(data: _tany, dtype: _odtype = None) -> _tarray:

    if is_list(data):
        output = _np.array(data)
    elif is_array(data):
        output = _np.copy(data)
    elif is_spmatrix(data):
        output = _np.array(data.todense())
    elif is_pandas(data):
        output = data.to_numpy(copy=True)
    else:
        output = None

    if output is None or not _np.issubdtype(output.dtype, _np.number):
        raise TypeError('The data type is not supported.')

    if dtype is not None:
        output = output.astype(dtype)

    return output


def get_caller(stack: _tstack) -> str:

    caller = stack[1][3]

    return caller


def get_file_extension(file_path: _tpath) -> str:

    if isinstance(file_path, str):
        file_path = _pl.Path(file_path)

    file_extension = ''.join(file_path.suffixes).lower()

    return file_extension


def get_instance_generators(cls: _tany) -> _tlist_str:

    instance_generators = []

    if cls is not None:
        for member_name, member in _ins.getmembers(cls, predicate=_ins.isfunction):
            if member_name[0] != '_' and hasattr(member, '_instance_generator'):
                instance_generators.append(member_name)

    return instance_generators


# noinspection PyBroadException
def get_full_name(obj: _tany) -> str:

    try:
        module = obj.__module__
    except Exception:
        module = obj.__class__.__module__

    try:
        name = obj.__qualname__
    except Exception:
        name = obj.__class__.__qualname__

    if module is None or module == 'builtins':
        full_name = name
    else:
        full_name = f'{module}.{name}'

    return full_name


def get_numpy_random_distributions() -> _tlist_str:

    try:
        from numpydoc.docscrape import NumpyDocString
    except ImportError:  # pragma: no cover
        return []

    excluded_funcs = ('dirichlet', 'multinomial', 'multivariate_normal')
    valid_summaries = ('DRAW RANDOM SAMPLES', 'DRAW SAMPLES', 'DRAWS SAMPLES')

    random_distributions = []

    for func_name in dir(_npr.RandomState):  # pylint: disable=no-member

        func = getattr(_npr.RandomState, func_name)  # pylint: disable=no-member

        if not callable(func) or func_name.startswith('_') or func_name.startswith('standard_') or func_name in excluded_funcs:
            continue

        doc = NumpyDocString(func.__doc__)

        if 'Summary' not in doc:
            continue

        doc_summary = doc['Summary']

        if not isinstance(doc_summary, list) or (len(doc_summary) == 0) or not isinstance(doc_summary[0], str):
            continue

        doc_summary_first = doc_summary[0].upper()

        for valid_summary in valid_summaries:
            if doc_summary_first.startswith(valid_summary):
                random_distributions.append(func_name)
                break

    return random_distributions


def is_array(value: _tany) -> bool:

    return isinstance(value, _np.ndarray)


def is_boolean(value: _tany) -> bool:

    return isinstance(value, bool)


def is_dictionary(value: _tany) -> bool:

    return isinstance(value, dict)


def is_float(value: _tany) -> bool:

    return isinstance(value, (float, _np.floating))


def is_graph(value: _tany) -> _tpair_bool:

    if isinstance(value, _nx.MultiDiGraph):
        return True, True

    if isinstance(value, _nx.DiGraph):
        return True, False

    return False, False


def is_integer(value: _tany) -> bool:

    return isinstance(value, (int, _np.integer)) and not isinstance(value, bool)


def is_list(value: _tany) -> bool:

    return isinstance(value, list)


def is_number(value: _tany) -> bool:

    return is_float(value) or is_integer(value)


def is_pandas(value: _tany) -> bool:

    return _pandas_found and isinstance(value, (_pd.DataFrame, _pd.Series))


def is_spmatrix(value: _tany) -> bool:

    return isinstance(value, _spsp.spmatrix)


def is_string(value: _tany) -> bool:

    return isinstance(value, str) and len(value.strip()) > 0


def is_tuple(value: _tany) -> bool:

    return isinstance(value, tuple)
