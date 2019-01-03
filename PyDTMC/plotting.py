# -*- coding: utf-8 -*-

__all__ = [
    'plot_distribution', 'plot_eigenvalues', 'plot_graph', 'plot_walk'
]


###########
# IMPORTS #
###########


import inspect as ip
import io as io
import matplotlib.colors as mlc
import matplotlib.image as mli
import matplotlib.pyplot as mlp
import matplotlib.ticker as mlt
import networkx as nx
import numpy as np
import numpy.linalg as npl
import subprocess as sp

from globals import *
from markov_chain import *
from validation import *


#############
# FUNCTIONS #
#############


def plot_distribution(mc: MarkovChain, distribution: tdistribution, plot_type: str = 'curves') -> oplot:

    if not isinstance(mc, MarkovChain):
        raise ValidationError('A valid MarkovChain instance must be provided.')

    try:

        distribution = validate_distribution(distribution, mc.size)
        plot_type = validate_enumerator(plot_type, ['curves', 'heatmap'])

    except Exception as e:
        argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
        raise ValidationError(str(e).replace('@arg@', argument))

    if isinstance(distribution, int):
        distribution = mc.redistribute(distribution, include_initial=True)

    distribution_len = len(distribution)
    distribution = np.array(distribution)

    figure, ax = mlp.subplots(dpi=dpi)

    if plot_type == 'curves':

        ax.set_prop_cycle('color', colors)

        for i in range(mc.size):
            ax.plot(np.arange(0.0, distribution_len, 1.0), distribution[:, i], label=mc.states[i], marker='o')

        if np.array_equal(distribution[0, :], np.ones(mc.size, dtype=float) / mc.size):
            ax.plot(0.0, distribution[0, 0], color=color_darkest, label="Start", marker='o', markeredgecolor=color_darkest, markerfacecolor=color_darkest)
            legend_size = mc.size + 1
        else:
            legend_size = mc.size

        ax.set_xlabel('Steps', fontsize=13.0)
        ax.set_xticks(np.arange(0.0, distribution_len, 1.0 if distribution_len <= 11 else 10.0))
        ax.set_xlim(-0.5, distribution_len - 0.5)

        ax.set_ylabel('Frequencies', fontsize=13.0)
        ax.set_yticks(np.linspace(0.0, 1.0, 11))
        ax.set_ylim(0.0, 1.0)

        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=legend_size)
        ax.set_title('Distplot (Curves)', fontsize=15.0, fontweight='bold')

        mlp.subplots_adjust(bottom=0.2)

    else:

        color_map = mlc.LinearSegmentedColormap.from_list('ColorMap', [color_brightest, colors[0]], 20)
        ax_is = ax.imshow(np.transpose(distribution), aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        ax.set_xlabel('Steps', fontsize=13.0)
        ax.set_xticks(np.arange(0.0, distribution_len + 1.0, 1.0 if distribution_len <= 11 else 10.0), minor=False)
        ax.set_xticks(np.arange(-0.5, distribution_len, 1.0), minor=True)
        ax.set_xticklabels(np.arange(0.0, distribution_len, 1.0 if distribution_len <= 11 else 10.0))
        ax.set_xlim(-0.5, distribution_len - 0.5)

        ax.set_yticks(np.arange(0.0, mc.size, 1.0), minor=False)
        ax.set_yticks(np.arange(-0.5, mc.size, 1.0), minor=True)
        ax.set_yticklabels(mc.states)

        ax.grid(which='minor', color='k')

        cb = figure.colorbar(ax_is, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

        ax.set_title('Distplot (Heatmap)', fontsize=15.0, fontweight='bold')

    if mlp.isinteractive():
        mlp.show(block=False)
        return None

    return figure, ax


def plot_eigenvalues(mc: MarkovChain) -> oplot:

    if not isinstance(mc, MarkovChain):
        raise ValidationError('A valid MarkovChain instance must be provided.')

    figure, ax = mlp.subplots(dpi=dpi)

    handles = list()
    labels = list()

    theta = np.linspace(0.0, 2.0 * np.pi, 200)

    values, _ = npl.eig(mc.p)
    values = values.astype(complex)
    values_final = np.unique(np.append(values, np.array([1.0]).astype(complex)))

    x_unit_circle = np.cos(theta)
    y_unit_circle = np.sin(theta)

    if mc.is_ergodic:

        values_abs = np.sort(np.abs(values))
        values_ct1 = np.isclose(values_abs, 1.0)

        if not np.all(values_ct1):

            mu = values_abs[~values_ct1][-1]

            if not np.isclose(mu, 0.0):

                x_slem_circle = mu * x_unit_circle
                y_slem_circle = mu * y_unit_circle

                cs = np.linspace(-1.1, 1.1, 201)
                x_spectral_gap, y_spectral_gap = np.meshgrid(cs, cs)
                z_spectral_gap = x_spectral_gap ** 2 + y_spectral_gap ** 2

                h = ax.contourf(x_spectral_gap, y_spectral_gap, z_spectral_gap, alpha=0.2, colors='r', levels=[mu ** 2.0, 1.0])
                handles.append(mlp.Rectangle((0.0, 0.0), 1.0, 1.0, fc=h.collections[0].get_facecolor()[0]))
                labels.append('Spectral Gap')

                ax.plot(x_slem_circle, y_slem_circle, color='red', linestyle='--', linewidth=1.5)

    ax.plot(x_unit_circle, y_unit_circle, color='red', linestyle='-', linewidth=3.0)

    h, = ax.plot(np.real(values_final), np.imag(values_final), color='blue', linestyle='None', marker='*', markersize=12.5)
    handles.append(h)
    labels.append('Eigenvalues')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    formatter = mlt.FormatStrFormatter('%g')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xticks(np.linspace(-1.0, 1.0, 9))
    ax.set_yticks(np.linspace(-1.0, 1.0, 9))
    ax.grid(which='major')

    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(handles))
    ax.set_title('Eigenplot', fontsize=15.0, fontweight='bold')

    mlp.subplots_adjust(bottom=0.2)

    if mlp.isinteractive():
        mlp.show(block=False)
        return None

    return figure, ax


def plot_graph(mc: MarkovChain, nodes_color: bool = True, nodes_type: bool = True, edges_color: bool = True, edges_value: bool = True) -> oplot:

    def edge_colors(hex_from: str, hex_to: str, steps: int) -> lstr:

        begin = [int(hex_from[i:i + 2], 16) for i in range(1, 6, 2)]
        end = [int(hex_to[i:i + 2], 16) for i in range(1, 6, 2)]

        clist = [hex_from]

        for s in range(1, steps):
            vector = [int(begin[j] + (float(s) / (steps - 1)) * (end[j] - begin[j])) for j in range(3)]
            rgb = [int(v) for v in vector]
            clist.append(f'#{"".join(["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in rgb])}')

        return clist

    def node_colors(count: int) -> lstr:

        colors_limit = len(colors) - 1
        offset = 0

        clist = list()

        while count > 0:
            clist.append(colors[offset])
            offset += 1
            if offset > colors_limit:
                offset = 0
            count -= 1

        return clist

    try:
        sp.call(['dot', '-V'], stdout=sp.PIPE, stderr=sp.PIPE)
    except Exception:
        raise EnvironmentError('Graphviz is required by this plotting function.')

    try:
        import pydot as pyd
    except ImportError:
        raise ImportError('Pydot is required by this plotting function.')

    if not isinstance(mc, MarkovChain):
        raise ValidationError('A valid MarkovChain instance must be provided.')

    try:

        nodes_color = validate_boolean(nodes_color)
        nodes_type = validate_boolean(nodes_type)
        edges_color = validate_boolean(edges_color)
        edges_value = validate_boolean(edges_value)

    except Exception as e:
        argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
        raise ValidationError(str(e).replace('@arg@', argument))

    g = mc.to_directed_graph()
    g_pydot = nx.nx_pydot.to_pydot(g)

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
        c = edge_colors(color_gray, color_darkest, 20)
        for edge in g_pydot.get_edges():
            probability = mc.transition_probability(edge.get_source(), edge.get_destination())
            x = int(round(probability * 20.0)) - 1
            edge.set_style('filled')
            edge.set_color(c[x])

    if edges_value:
        for edge in g_pydot.get_edges():
            probability = mc.transition_probability(edge.get_source(), edge.get_destination())
            if probability.is_integer():
                edge.set_label(f' {round(probability,2):g}.0 ')
            else:
                edge.set_label(f' {round(probability,2):g} ')

    buffer = io.BytesIO()
    buffer.write(g_pydot.create_png())
    buffer.seek(0)

    img = mli.imread(buffer)
    img_x = img.shape[0] / dpi
    img_y = img.shape[1] / dpi

    figure = mlp.figure(figsize=(img_y, img_x), dpi=dpi)
    figure.figimage(img)

    if mlp.isinteractive():
        mlp.show(block=False)
        return None

    return figure, figure.gca()


def plot_walk(mc: MarkovChain, walk: twalk, plot_type: str = 'histogram') -> oplot:

    if not isinstance(mc, MarkovChain):
        raise ValidationError('A valid MarkovChain instance must be provided.')

    try:

        walk = validate_walk(walk, mc.states)
        plot_type = validate_enumerator(plot_type, ['histogram', 'sequence', 'transitions'])

    except Exception as e:
        argument = ''.join(ip.trace()[0][4]).split('=', 1)[0].strip()
        raise ValidationError(str(e).replace('@arg@', argument))

    if isinstance(walk, int):
        walk = mc.walk(walk, include_initial=True, output_indices=True)

    walk_len = len(walk)

    figure, ax = mlp.subplots(dpi=dpi)

    if plot_type == 'histogram':

        walk_histogram = np.zeros((mc.size, walk_len), dtype=float)

        for i, s in enumerate(walk):
            walk_histogram[s, i] = 1.0

        walk_histogram = np.sum(walk_histogram, axis=1) / np.sum(walk_histogram)

        ax.bar(np.arange(0.0, mc.size, 1.0), walk_histogram, edgecolor=color_darkest, facecolor=colors[0])

        ax.set_xlabel('States', fontsize=13.0)
        ax.set_xticks(np.arange(0.0, mc.size, 1.0))
        ax.set_xticklabels(mc.states)

        ax.set_ylabel('Frequencies', fontsize=13.0)
        ax.set_yticks(np.linspace(0.0, 1.0, 11))
        ax.set_ylim(0.0, 1.0)

        ax.set_title('Walkplot (Histogram)', fontsize=15.0, fontweight='bold')

    elif plot_type == 'sequence':

        walk_sequence = np.zeros((mc.size, walk_len), dtype=float)

        for i, s in enumerate(walk):
            walk_sequence[s, i] = 1.0

        color_map = mlc.LinearSegmentedColormap.from_list('ColorMap', [color_brightest, colors[0]], 2)
        ax.imshow(walk_sequence, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        ax.set_xlabel('Steps', fontsize=13.0)
        ax.set_xticks(np.arange(0.0, walk_len + 1.0, 1.0 if walk_len <= 11 else 10.0), minor=False)
        ax.set_xticks(np.arange(-0.5, walk_len, 1.0), minor=True)
        ax.set_xticklabels(np.arange(0.0, walk_len, 1.0 if walk_len <= 11 else 10.0))
        ax.set_xlim(-0.5, walk_len - 0.5)

        ax.set_ylabel('States', fontsize=13.0)
        ax.set_yticks(np.arange(0.0, mc.size, 1.0), minor=False)
        ax.set_yticks(np.arange(-0.5, mc.size, 1.0), minor=True)
        ax.set_yticklabels(mc.states)

        ax.grid(which='minor', color='k')

        ax.set_title('Walkplot (Sequence)', fontsize=15.0, fontweight='bold')

    else:

        walk_transitions = np.zeros((mc.size, mc.size), dtype=float)

        for i in range(1, walk_len):
            walk_transitions[walk[i - 1], walk[i]] += 1.0

        walk_transitions = walk_transitions / np.sum(walk_transitions)

        color_map = mlc.LinearSegmentedColormap.from_list('ColorMap', [color_brightest, colors[0]], 20)
        ax_is = ax.imshow(walk_transitions, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        ax.set_xticks(np.arange(0.0, mc.size, 1.0), minor=False)
        ax.set_xticks(np.arange(-0.5, mc.size, 1.0), minor=True)
        ax.set_xticklabels(mc.states)

        ax.set_yticks(np.arange(0.0, mc.size, 1.0), minor=False)
        ax.set_yticks(np.arange(-0.5, mc.size, 1.0), minor=True)
        ax.set_yticklabels(mc.states)

        ax.grid(which='minor', color='k')

        cb = figure.colorbar(ax_is, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

        ax.set_title('Walkplot (Transitions)', fontsize=15.0, fontweight='bold')

    if mlp.isinteractive():
        mlp.show(block=False)
        return None

    return figure, ax
