# -*- coding: utf-8 -*-

__all__ = [
    'plot_eigenvalues', 'plot_graph', 'plot_redistributions', 'plot_walk'
]


###########
# IMPORTS #
###########


# Major

import matplotlib.colors as _mc
import matplotlib.image as _mi
import matplotlib.pyplot as _mp
import matplotlib.ticker as _mt
import networkx as _nx
import numpy as _np
import numpy.linalg as _npl

# Minor

from inspect import (
    trace as _trace
)

from io import (
    BytesIO as _BytesIO
)

from subprocess import (
    call as _call,
    PIPE as _PIPE
)

from typing import (
    Iterable as _Iterable,
    List as _List,
    Optional as _Optional,
    Tuple as _Tuple,
    Union as _Union
)

# Internal

from pydtmc.custom_types import (
    tnumeric as _tnumeric
)

from pydtmc.exceptions import (
    ValidationError as _ValidationError
)

from pydtmc.markov_chain import (
    MarkovChain as _MarkovChain
)

from pydtmc.validation import (
    validate_boolean as _validate_boolean,
    validate_distribution as _validate_distribution,
    validate_enumerator as _validate_enumerator,
    validate_dpi as _validate_dpi,
    validate_state as _validate_state,
    validate_status as _validate_status,
    validate_walk as _validate_walk
)


##############
# ATTRIBUTES #
##############


_color_black = '#000000'
_color_gray = '#E0E0E0'
_color_white = '#FFFFFF'
_colors = ['#80B1D3', '#FFED6F', '#B3DE69', '#BEBADA', '#FDB462', '#8DD3C7', '#FB8072', '#FCCDE5']


#############
# FUNCTIONS #
#############


def plot_eigenvalues(mc: _MarkovChain, dpi: int = 100) -> _Optional[_Tuple[_mp.Figure, _mp.Axes]]:

    """
    The function plots the eigenvalues of the Markov chain on the complex plane.

    :param mc: the target Markov chain.
    :param dpi: the resolution of the plot expressed in dots per inch (by default, 100).
    :return: None if Matplotlib is in interactive mode as the plot is immediately displayed, otherwise the handles of the plot.
    :raises ValidationError: if any input argument is not compliant.
    """

    if not isinstance(mc, _MarkovChain):
        raise ValueError('A valid MarkovChain instance must be provided.')

    try:

        dpi = _validate_dpi(dpi)

    except Exception as e:
        argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
        raise _ValidationError(str(e).replace('@arg@', argument)) from None

    figure, ax = _mp.subplots(dpi=dpi)

    handles = list()
    labels = list()

    theta = _np.linspace(0.0, 2.0 * _np.pi, 200)

    values, _ = _npl.eig(mc.p)
    values = values.astype(complex)
    values_final = _np.unique(_np.append(values, _np.array([1.0]).astype(complex)))

    x_unit_circle = _np.cos(theta)
    y_unit_circle = _np.sin(theta)

    if mc.is_ergodic:

        values_abs = _np.sort(_np.abs(values))
        values_ct1 = _np.isclose(values_abs, 1.0)

        if not _np.all(values_ct1):

            mu = values_abs[~values_ct1][-1]

            if not _np.isclose(mu, 0.0):

                x_slem_circle = mu * x_unit_circle
                y_slem_circle = mu * y_unit_circle

                cs = _np.linspace(-1.1, 1.1, 201)
                x_spectral_gap, y_spectral_gap = _np.meshgrid(cs, cs)
                z_spectral_gap = x_spectral_gap ** 2 + y_spectral_gap ** 2

                h = ax.contourf(x_spectral_gap, y_spectral_gap, z_spectral_gap, alpha=0.2, colors='r', levels=[mu ** 2.0, 1.0])
                handles.append(_mp.Rectangle((0.0, 0.0), 1.0, 1.0, fc=h.collections[0].get_facecolor()[0]))
                labels.append('Spectral Gap')

                ax.plot(x_slem_circle, y_slem_circle, color='red', linestyle='--', linewidth=1.5)

    ax.plot(x_unit_circle, y_unit_circle, color='red', linestyle='-', linewidth=3.0)

    h, = ax.plot(_np.real(values_final), _np.imag(values_final), color='blue', linestyle='None', marker='*', markersize=12.5)
    handles.append(h)
    labels.append('Eigenvalues')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    formatter = _mt.FormatStrFormatter('%g')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xticks(_np.linspace(-1.0, 1.0, 9))
    ax.set_yticks(_np.linspace(-1.0, 1.0, 9))
    ax.grid(which='major')

    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(handles))
    ax.set_title('Eigenplot', fontsize=15.0, fontweight='bold')

    _mp.subplots_adjust(bottom=0.2)

    if _mp.isinteractive():
        _mp.show(block=False)
        return None

    return figure, ax


def plot_graph(mc: _MarkovChain, nodes_color: bool = True, nodes_type: bool = True, edges_color: bool = True, edges_value: bool = True, dpi: int = 100) -> _Optional[_Tuple[_mp.Figure, _mp.Axes]]:

    """
    The function plots the directed graph of the Markov chain.

    | **Notes:** Graphviz and Pydot are not required, but they provide access to extended graphs with additional features.

    :param mc: the target Markov chain.
    :param nodes_color: a boolean indicating whether to display colored nodes based on communicating classes (by default, True).
    :param nodes_type: a boolean indicating whether to use a different shape for every node type (by default, True).
    :param edges_color: a boolean indicating whether to display edges using a gradient based on transition probabilities, valid only for extended graphs (by default, True).
    :param edges_value: a boolean indicating whether to display the transition probability of every edge (by default, True).
    :param dpi: the resolution of the plot expressed in dots per inch (by default, 100).
    :return: None if Matplotlib is in interactive mode as the plot is immediately displayed, otherwise the handles of the plot.
    :raises ValidationError: if any input argument is not compliant.
    """

    def edge_colors(hex_from: str, hex_to: str, steps: int) -> _List[str]:

        begin = [int(hex_from[i:i + 2], 16) for i in range(1, 6, 2)]
        end = [int(hex_to[i:i + 2], 16) for i in range(1, 6, 2)]

        clist = [hex_from]

        for s in range(1, steps):
            vector = [int(begin[j] + (float(s) / (steps - 1)) * (end[j] - begin[j])) for j in range(3)]
            rgb = [int(v) for v in vector]
            clist.append(f'#{"".join(["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in rgb])}')

        return clist

    def node_colors(count: int) -> _List[str]:

        colors_limit = len(_colors) - 1
        offset = 0

        clist = list()

        while count > 0:
            clist.append(_colors[offset])
            offset += 1
            if offset > colors_limit:
                offset = 0
            count -= 1

        return clist

    if not isinstance(mc, _MarkovChain):
        raise ValueError('A valid MarkovChain instance must be provided.')

    try:

        nodes_color = _validate_boolean(nodes_color)
        nodes_type = _validate_boolean(nodes_type)
        edges_color = _validate_boolean(edges_color)
        edges_value = _validate_boolean(edges_value)
        dpi = _validate_dpi(dpi)

    except Exception as e:
        argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
        raise _ValidationError(str(e).replace('@arg@', argument)) from None

    extended_graph = True

    # noinspection PyBroadException
    try:
        _call(['dot', '-V'], stdout=_PIPE, stderr=_PIPE)
    except Exception:
        extended_graph = False
        pass

    try:
        import pydot as pyd
    except ImportError:
        extended_graph = False
        pass

    g = mc.to_directed_graph()

    if extended_graph:

        g_pydot = _nx.nx_pydot.to_pydot(g)

        if nodes_color:
            c = node_colors(len(mc.communicating_classes))
            for node in g_pydot.get_nodes():
                state = node.get_name()
                for x, cc in enumerate(mc.communicating_classes):
                    if state in cc:
                        node.set_style('filled')
                        node.set_fillcolor(c[x])
                        break

        if nodes_type:
            for node in g_pydot.get_nodes():
                if node.get_name() in mc.transient_states:
                    node.set_shape('box')
                else:
                    node.set_shape('ellipse')

        if edges_color:
            c = edge_colors(_color_gray, _color_black, 20)
            for edge in g_pydot.get_edges():
                probability = mc.transition_probability(edge.get_destination(), edge.get_source())
                x = int(round(probability * 20.0)) - 1
                edge.set_style('filled')
                edge.set_color(c[x])

        if edges_value:
            for edge in g_pydot.get_edges():
                probability = mc.transition_probability(edge.get_destination(), edge.get_source())
                if probability.is_integer():
                    edge.set_label(f' {probability:g}.0 ')
                else:
                    edge.set_label(f' {round(probability,2):g} ')

        buffer = _BytesIO()
        buffer.write(g_pydot.create_png())
        buffer.seek(0)

        img = _mi.imread(buffer)
        img_x = img.shape[0] / dpi
        img_y = img.shape[1] / dpi

        figure = _mp.figure(figsize=(img_y, img_x), dpi=dpi)
        figure.figimage(img)
        ax = figure.gca()

    else:

        mpi = _mp.isinteractive()
        _mp.interactive(False)

        figure, ax = _mp.subplots(dpi=dpi)

        positions = _nx.spring_layout(g)
        node_colors_all = node_colors(len(mc.communicating_classes))

        for node in g.nodes:

            node_color = None

            if nodes_color:
                for x, cc in enumerate(mc.communicating_classes):
                    if node in cc:
                        node_color = node_colors_all[x]
                        break

            if nodes_type:
                if node in mc.transient_states:
                    node_shape = 's'
                else:
                    node_shape = 'o'
            else:
                node_shape = None

            if node_color is not None and node_shape is not None:
                _nx.draw_networkx_nodes(g, positions, ax=ax, nodelist=[node], edgecolors='k', node_color=node_color, node_shape=node_shape)
            elif node_color is not None and node_shape is None:
                _nx.draw_networkx_nodes(g, positions, ax=ax, nodelist=[node], edgecolors='k', node_color=node_color)
            elif node_color is None and node_shape is not None:
                _nx.draw_networkx_nodes(g, positions, ax=ax, nodelist=[node], edgecolors='k', node_shape=node_shape)
            else:
                _nx.draw_networkx_nodes(g, positions, ax=ax, edgecolors='k')

        _nx.draw_networkx_labels(g, positions, ax=ax)

        _nx.draw_networkx_edges(g, positions, ax=ax, arrows=False)

        if edges_value:

            edges_values = dict()

            for edge in g.edges:
                probability = mc.transition_probability(edge[1], edge[0])
                if probability.is_integer():
                    value = f' {probability:g}.0 '
                else:
                    value = f' {round(probability,2):g} '
                edges_values[(edge[0], edge[1])] = value

            _nx.draw_networkx_edge_labels(g, positions, ax=ax, edge_labels=edges_values, label_pos=0.7)

        _mp.interactive(mpi)

    if _mp.isinteractive():
        _mp.show(block=False)
        return None

    return figure, ax


def plot_redistributions(mc: _MarkovChain, distributions: _Union[int, _Iterable[_np.ndarray]], initial_status: _Optional[_Union[int, str, _tnumeric]] = None, plot_type: str = 'projection', dpi: int = 100) -> _Optional[_Tuple[_mp.Figure, _mp.Axes]]:

    """
    The function plots a redistribution of states on the given Markov chain.

    :param mc: the target Markov chain.
    :param distributions: a sequence of redistributions or the number of redistributions to perform.
    :param initial_status: the initial state or the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
    :param plot_type: the type of plot to display (either heatmap or projection; projection by default).
    :param dpi: the resolution of the plot expressed in dots per inch (by default, 100).
    :return: None if Matplotlib is in interactive mode as the plot is immediately displayed, otherwise the handles of the plot.
    :raises ValidationError: if any input argument is not compliant.
    """

    if not isinstance(mc, _MarkovChain):
        raise ValueError('A valid MarkovChain instance must be provided.')

    try:

        distributions = _validate_distribution(distributions, mc.size)

        if initial_status is not None:
            initial_status = _validate_status(initial_status, mc.states)

        plot_type = _validate_enumerator(plot_type, ['heatmap', 'projection'])
        dpi = _validate_dpi(dpi)

    except Exception as e:
        argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
        raise _ValidationError(str(e).replace('@arg@', argument)) from None

    if isinstance(distributions, int):
        distributions = mc.redistribute(distributions, initial_status=initial_status, include_initial=True, output_last=False)
    elif initial_status is not None and not _np.array_equal(distributions[0], initial_status):
        raise ValueError('The "initial_status" parameter, if specified when the "distributions" parameter represents a sequence of redistributions, must match the first element.')

    distribution_len = len(distributions)
    distributions = _np.array(distributions)

    figure, ax = _mp.subplots(dpi=dpi)

    if plot_type == 'heatmap':

        color_map = _mc.LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 20)
        ax_is = ax.imshow(_np.transpose(distributions), aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        ax.set_xlabel('Steps', fontsize=13.0)
        ax.set_xticks(_np.arange(0.0, distribution_len + 1.0, 1.0 if distribution_len <= 11 else 10.0), minor=False)
        ax.set_xticks(_np.arange(-0.5, distribution_len, 1.0), minor=True)
        ax.set_xticklabels(_np.arange(0, distribution_len, 1 if distribution_len <= 11 else 10))
        ax.set_xlim(-0.5, distribution_len - 0.5)

        ax.set_yticks(_np.arange(0.0, mc.size, 1.0), minor=False)
        ax.set_yticks(_np.arange(-0.5, mc.size, 1.0), minor=True)
        ax.set_yticklabels(mc.states)

        ax.grid(which='minor', color='k')

        cb = figure.colorbar(ax_is, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

        ax.set_title('Redistplot (Heatmap)', fontsize=15.0, fontweight='bold')

    else:

        ax.set_prop_cycle('color', _colors)

        if distribution_len == 2:
            for i in range(mc.size):
                ax.plot(_np.arange(0.0, distribution_len, 1.0), distributions[:, i], label=mc.states[i], marker='o')
        else:
            for i in range(mc.size):
                ax.plot(_np.arange(0.0, distribution_len, 1.0), distributions[:, i], label=mc.states[i])

        if _np.array_equal(distributions[0, :], _np.ones(mc.size, dtype=float) / mc.size):
            ax.plot(0.0, distributions[0, 0], color=_color_black, label="Start", marker='o', markeredgecolor=_color_black, markerfacecolor=_color_black)
            legend_size = mc.size + 1
        else:
            legend_size = mc.size

        ax.set_xlabel('Steps', fontsize=13.0)
        ax.set_xticks(_np.arange(0.0, distribution_len, 1.0 if distribution_len <= 11 else 10.0))
        ax.set_xticklabels(_np.arange(0, distribution_len, 1 if distribution_len <= 11 else 10))
        ax.set_xlim(-1.0 * (distribution_len * 0.05), distribution_len * 1.05)

        ax.set_ylabel('Frequencies', fontsize=13.0)
        ax.set_yticks(_np.linspace(0.0, 1.0, 11))
        ax.set_ylim(-0.05, 1.05)

        ax.grid()
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=legend_size)
        ax.set_title('Redistplot (Projection)', fontsize=15.0, fontweight='bold')

        _mp.subplots_adjust(bottom=0.2)

    if _mp.isinteractive():
        _mp.show(block=False)
        return None

    return figure, ax


def plot_walk(mc: _MarkovChain, walk: _Union[int, _Iterable[int], _Iterable[str]], initial_state: _Optional[_Union[int, str]] = None, plot_type: str = 'histogram', dpi: int = 100) -> _Optional[_Tuple[_mp.Figure, _mp.Axes]]:

    """
    The function plots a random walk on the given Markov chain.

    :param mc: the target Markov chain.
    :param walk: a sequence of states or the number of simulations to perform.
    :param initial_state: the initial state of the walk (if omitted, it is chosen uniformly at random).
    :param plot_type: the type of plot to display (one of histogram, sequence and transitions; histogram by default).
    :param dpi: the resolution of the plot expressed in dots per inch (by default, 100).
    :return: None if Matplotlib is in interactive mode as the plot is immediately displayed, otherwise the handles of the plot.
    :raises ValidationError: if any input argument is not compliant.
    """

    if not isinstance(mc, _MarkovChain):
        raise ValueError('A valid MarkovChain instance must be provided.')

    try:

        walk = _validate_walk(walk, mc.states)

        if initial_state is not None:
            initial_state = _validate_state(initial_state, mc.states)

        plot_type = _validate_enumerator(plot_type, ['histogram', 'sequence', 'transitions'])
        dpi = _validate_dpi(dpi)

    except Exception as e:
        argument = ''.join(_trace()[0][4]).split('=', 1)[0].strip()
        raise _ValidationError(str(e).replace('@arg@', argument)) from None

    if isinstance(walk, int):
        walk = mc.walk(walk, initial_state=initial_state, include_initial=True, output_indices=True)
    elif initial_state is not None and (walk[0] != initial_state):
        raise ValueError('The "initial_state" parameter, if specified when the "walk" parameter represents a sequence of states, must match the first element.')

    walk_len = len(walk)

    figure, ax = _mp.subplots(dpi=dpi)

    if plot_type == 'histogram':

        walk_histogram = _np.zeros((mc.size, walk_len), dtype=float)

        for i, s in enumerate(walk):
            walk_histogram[s, i] = 1.0

        walk_histogram = _np.sum(walk_histogram, axis=1) / _np.sum(walk_histogram)

        ax.bar(_np.arange(0.0, mc.size, 1.0), walk_histogram, edgecolor=_color_black, facecolor=_colors[0])

        ax.set_xlabel('States', fontsize=13.0)
        ax.set_xticks(_np.arange(0.0, mc.size, 1.0))
        ax.set_xticklabels(mc.states)

        ax.set_ylabel('Frequencies', fontsize=13.0)
        ax.set_yticks(_np.linspace(0.0, 1.0, 11))
        ax.set_ylim(0.0, 1.0)

        ax.set_title('Walkplot (Histogram)', fontsize=15.0, fontweight='bold')

    elif plot_type == 'sequence':

        walk_sequence = _np.zeros((mc.size, walk_len), dtype=float)

        for i, s in enumerate(walk):
            walk_sequence[s, i] = 1.0

        color_map = _mc.LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 2)
        ax.imshow(walk_sequence, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        ax.set_xlabel('Steps', fontsize=13.0)
        ax.set_xticks(_np.arange(0.0, walk_len + 1.0, 1.0 if walk_len <= 11 else 10.0), minor=False)
        ax.set_xticks(_np.arange(-0.5, walk_len, 1.0), minor=True)
        ax.set_xticklabels(_np.arange(0, walk_len, 1 if walk_len <= 11 else 10))
        ax.set_xlim(-0.5, walk_len - 0.5)

        ax.set_ylabel('States', fontsize=13.0)
        ax.set_yticks(_np.arange(0.0, mc.size, 1.0), minor=False)
        ax.set_yticks(_np.arange(-0.5, mc.size, 1.0), minor=True)
        ax.set_yticklabels(mc.states)

        ax.grid(which='minor', color='k')

        ax.set_title('Walkplot (Sequence)', fontsize=15.0, fontweight='bold')

    else:

        walk_transitions = _np.zeros((mc.size, mc.size), dtype=float)

        for i in range(1, walk_len):
            walk_transitions[walk[i - 1], walk[i]] += 1.0

        walk_transitions = walk_transitions / _np.sum(walk_transitions)

        color_map = _mc.LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 20)
        ax_is = ax.imshow(walk_transitions, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        ax.set_xticks(_np.arange(0.0, mc.size, 1.0), minor=False)
        ax.set_xticks(_np.arange(-0.5, mc.size, 1.0), minor=True)
        ax.set_xticklabels(mc.states)

        ax.set_yticks(_np.arange(0.0, mc.size, 1.0), minor=False)
        ax.set_yticks(_np.arange(-0.5, mc.size, 1.0), minor=True)
        ax.set_yticklabels(mc.states)

        ax.grid(which='minor', color='k')

        cb = figure.colorbar(ax_is, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

        ax.set_title('Walkplot (Transitions)', fontsize=15.0, fontweight='bold')

    if _mp.isinteractive():
        _mp.show(block=False)
        return None

    return figure, ax
