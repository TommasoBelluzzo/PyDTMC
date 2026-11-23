# -*- coding: utf-8 -*-

__all__ = [
    'plot_hmm_extended',
    'plot_hmm_standard',
    'plot_mc_extended',
    'plot_mc_standard',
    'plot_sequence_heatmap',
    'plot_sequence_histogram',
    'plot_sequence_matrix',
    'plot_trellis_extended',
    'plot_trellis_standard'
]


###########
# IMPORTS #
###########

# Standard

import copy as _cp
import io as _io
import math as _mt

# Libraries

import matplotlib.colorbar as _mplcb
import matplotlib.colors as _mplcr
import matplotlib.image as _mpli
import matplotlib.patches as _mplpc
import matplotlib.pyplot as _mplp
import networkx as _nx
import numpy as _np
import numpy.linalg as _npl

try:
    import pydot as _pyd
    _pydot_found = True
except ImportError:  # pragma: no cover
    _pyd = None
    _pydot_found = False


#############
# CONSTANTS #
#############

_color_black = '#000000'
_color_gray = '#E0E0E0'
_color_white = '#FFFFFF'
_colors = ('#80B1D3', '#FFED6F', '#B3DE69', '#BEBADA', '#FDB462', '#8DD3C7', '#FB8072', '#FCCDE5', '#E5C494')

_default_color_edge = _color_black
_default_color_node = _color_white
_default_color_path = _colors[0]
_default_color_symbol = _color_gray
_default_node_size = 600


#############
# FUNCTIONS #
#############


def _decode_image(g, dpi):

    buffer = _io.BytesIO()
    buffer.write(g.create(format='png'))
    buffer.seek(0)

    img = _mpli.imread(buffer)

    img_x = img.shape[0] / dpi
    img_xi = img_x * 1.1
    img_xo = ((img_xi - img_x) / 2.0) * dpi

    img_y = img.shape[1] / dpi
    img_yi = img_y * 1.1
    img_yo = ((img_yi - img_y) / 2.0) * dpi

    return img, img_x, img_xo, img_y, img_yo


def _xticks_labels(ax, size, labels_name, labels, minor_major):

    if labels_name is not None:
        ax.set_xlabel(labels_name, fontsize=13.0)

    if minor_major:
        ax.set_xticks(_np.arange(0.0, size, 1.0), minor=False)
        ax.set_xticks(_np.arange(-0.5, size, 1.0), minor=True)
    else:
        ax.set_xticks(_np.arange(0.0, size, 1.0))

    ax.set_xticklabels(labels)


def _xticks_steps(ax, length, interval=1, max_ticks=11):

    ax.set_xlabel('Steps', fontsize=13.0)

    tick_step = 1
    if length > max_ticks:

        target_step = (length - 1) / (max_ticks - 1)
        magnitude = 10.0**_mt.floor(_mt.log10(target_step))

        for multiplier in (1, 2, 5, 10):
            tick_step = int(multiplier * magnitude)
            if tick_step >= target_step:
                break

    ticks = _np.arange(0, length, tick_step)

    ax.set_xticks(ticks, minor=False)
    ax.set_xticks(_np.arange(-0.5, length, 1.0), minor=True)
    ax.set_xticklabels(ticks * interval)
    ax.set_xlim(-0.5, length - 0.5)


def _yticks_frequency(ax, bottom, top):

    ax.set_ylabel('Frequency', fontsize=13.0)
    ax.set_yticks(_np.linspace(0.0, 1.0, 11))
    ax.set_ylim(bottom, top)


def _yticks_labels(ax, size, labels_name, labels):

    if labels_name is not None:
        ax.set_ylabel(labels_name, fontsize=13.0)

    ax.set_yticks(_np.arange(0.0, size, 1.0), minor=False)
    ax.set_yticks(_np.arange(-0.5, size, 1.0), minor=True)
    ax.set_yticklabels(labels)


def _get_boundaries(gb_d):

    i, j = gb_d.shape
    k = 0.1 / (i - 1.0) if i > 1 else 0.0

    b = _np.zeros((i, j), dtype=float)
    t = _np.zeros((i, j), dtype=float)

    for o in range(j):
        dj = gb_d[:, o]
        b[:, o] = _np.cumsum(dj + k) - dj - k
        t[:, o] = _np.cumsum(dj + k) - k

    b = _np.clip(b, 0.0, 1.0)
    t = _np.clip(t, 0.0, 1.0)

    return b, t


def _get_colors(gc_d):

    i = gc_d.shape[0]
    colors_count = len(_colors)

    colors = _np.zeros((i, 4), dtype=float)

    for index in range(i):

        base_color = _np.array(_mplcr.to_rgb(_colors[index % colors_count]))
        cycle = index // colors_count

        if cycle == 0:
            color = base_color
        else:  # pragma: no cover

            variation = min(0.12 * ((cycle + 1) // 2), 0.36)

            if (cycle % 2) == 1:
                color = base_color + ((1.0 - base_color) * variation)
            else:
                color = base_color * (1.0 - variation)

        colors[index, :3] = color
        colors[index, 3] = 1.0

    return colors


def _get_curves(gc_n, gc_x1, gc_y1, gc_x2, gc_y2):

    tx = _np.reshape(_np.linspace(gc_x1, gc_x2, 15), (1, -1))
    cx = _np.tile(_np.transpose(tx), (1, gc_n))

    ty = (1.0 - _np.cos(_np.reshape(_np.linspace(0.0, _np.pi, 15), (1, -1)))) / 2.0
    cy = _np.tile(gc_y1, (15, 1)) + (_np.tile(gc_y2 - gc_y1, (15, 1)) * _np.tile(_np.transpose(ty), (1, gc_n)))

    return cx, cy


def _get_legend(gl_mc, gl_c):

    handles = []
    labels = gl_mc.states

    for i, label in enumerate(labels):
        handles.append(_mplpc.Patch(color=gl_c[i, :], label=label))

    return handles, labels


def _get_polygons_bars(gpb_d, gpb_bb, gpb_bt, gpb_c):

    i, j = gpb_d.shape
    w = j / 40.0

    polygons = []

    for oj in range(j):

        xm = oj - w
        xp = oj + w

        for oi in range(i):

            yb = gpb_bb[oi, oj]
            yt = gpb_bt[oi, oj]

            x = [xm, xp, xp, xm]
            y = [yb, yb, yt, yt]

            polygons.append(_mplpc.Polygon(list(zip(x, y)), edgecolor=None, facecolor=gpb_c[oi, :], alpha=0.8))

    return polygons


def _get_polygons_flows(gpf_p, gpf_d, gpf_bb, gpf_c, gpf_interval):

    i, j = gpf_d.shape
    w = j / 40.0
    q = _npl.matrix_power(gpf_p, gpf_interval)

    polygons = []

    for oj in range(j - 1):

        bj = _np.copy(gpf_bb[:, oj + 1])

        x_lo = oj + w
        x_hi = oj - w + 1.0

        for oi in range(i):

            dij = gpf_d[oi, oj]
            bij = gpf_bb[oi, oj]

            qi = q[oi, :]
            qis = _np.cumsum(qi)

            tl = ((qis - qi) * dij) + bij
            bl = (qis * dij) + bij
            tr = _np.copy(bj)
            br = tr + bl - tl

            bj += bl - tl

            [bottom_x, bottom_y] = _get_curves(i, x_lo, bl, x_hi, br)
            [top_x, top_y] = _get_curves(i, x_hi, tr, x_lo, tl)
            x = _np.concatenate([bottom_x, top_x], axis=0)
            y = _np.concatenate([bottom_y, top_y], axis=0)

            for z in range(x.shape[1]):
                polygons.append(_mplpc.Polygon(list(zip(x[:, z], y[:, z])), edgecolor=None, facecolor=gpf_c[oi, :], alpha=0.3))

    return polygons


def _calculate_magnitude(*cm_elements):

    magnitudes = []

    for element in cm_elements:

        element_positive = element[element > 0.0]

        if element_positive.size > 0:
            element_minimum = _np.min(element_positive).item()
            element_magnitude = int(-_mt.floor(_mt.log10(element_minimum)))
            magnitudes.append(element_magnitude)

    magnitude = max(1, min(max(magnitudes), 4))

    return magnitude


def _draw_edge_labels_curved(delc_ax, delc_positions, delc_edge_labels):

    for (n1, n2), (rad, label) in delc_edge_labels.items():

        (x1, y1) = delc_positions[n1]
        (x2, y2) = delc_positions[n2]
        p1 = delc_ax.transData.transform(_np.array(delc_positions[n1]))
        p2 = delc_ax.transData.transform(_np.array(delc_positions[n2]))

        linear_mid = (0.5 * p1) + (0.5 * p2)
        cp_mid = linear_mid + (rad * _np.dot(_np.array([(0, 1), (-1, 0)]), p2 - p1))
        cp1 = (0.5 * p1) + (0.5 * cp_mid)
        cp2 = (0.5 * p2) + (0.5 * cp_mid)
        bezier_mid = (0.5 * cp1) + (0.5 * cp2)

        (x, y) = delc_ax.transData.inverted().transform(bezier_mid)
        xy = _np.array((x, y))

        angle = (_np.arctan2(y2 - y1, x2 - x1) / (2.0 * _np.pi)) * 360.0

        if angle > 90.0:
            angle -= 180.0

        if angle < -90.0:
            angle += 180.0

        rotation = delc_ax.transData.transform_angles(_np.array((angle,)), xy.reshape((1, 2)))[0]
        transform = delc_ax.transData
        bbox = {
            'boxstyle': 'round',
            'ec': (1.0, 1.0, 1.0),
            'fc': (1.0, 1.0, 1.0)
        }

        delc_ax.text(
            x, y,
            label, color='k', size=10, family='sans-serif', weight='normal',
            horizontalalignment='center', verticalalignment='center',
            bbox=bbox, clip_on=True, rotation=rotation, transform=transform, zorder=1
        )

    delc_ax.tick_params(
        axis='both', which='both',
        bottom=False, left=False,
        labelbottom=False, labelleft=False
    )


def _node_colors(nc_count):

    colors = _cp.deepcopy(_colors)
    colors_limit = len(colors) - 1
    colors_offset = 0

    colors_list = []

    while nc_count > 0:

        colors_list.append(colors[colors_offset])
        colors_offset += 1

        if colors_offset > colors_limit:  # pragma: no cover
            colors_offset = 0

        nc_count -= 1

    return colors_list



def _plot_graph_image(g, dpi):

    render_dpi = max(192, int(dpi * 2))
    g.set_dpi(str(render_dpi))
    img, img_height, _, img_width, _ = _decode_image(g, render_dpi)

    f, a = _mplp.subplots(figsize=(img_width * 1.1, img_height * 1.1), dpi=dpi)
    image_aspect = img.shape[1] / img.shape[0]
    artist = a.imshow(img, extent=(0.0, image_aspect, 0.0, 1.0), interpolation='hanning')
    a.axis('off')

    f.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)

    def _on_draw(event):

        nonlocal render_dpi

        if event.canvas is not f.canvas:
            return

        image_bounds = artist.get_window_extent(event.renderer)
        image_array = artist.get_array()
        scale = max(image_bounds.width / image_array.shape[1], image_bounds.height / image_array.shape[0])

        if scale <= 0.5:
            return

        next_dpi = int(_mt.ceil(render_dpi * scale * 2.2))
        g.set_dpi(str(next_dpi))

        try:
            image = _mpli.imread(_io.BytesIO(g.create(format='png')))
        except Exception:  # pragma: no cover
            g.set_dpi(str(render_dpi))
            return

        render_dpi = next_dpi
        artist.set_data(image)
        event.canvas.draw_idle()

    f.canvas.mpl_connect('draw_event', _on_draw)

    return f, a


def plot_hmm_extended(phe_hmm, phe_nodes_color, phe_nodes_type, phe_edges_label, phe_dpi):

    magnitude = _calculate_magnitude(phe_hmm.p, phe_hmm.e)

    node_colors = _node_colors(phe_hmm.n) if phe_nodes_color else []
    edge_colors = _cp.deepcopy(node_colors) if phe_nodes_color else []

    g = _pyd.Dot(graph_type='digraph')

    g_sub1 = _pyd.Subgraph()
    g.add_subgraph(g_sub1)

    g_sub2 = _pyd.Subgraph(rank='same')
    g.add_subgraph(g_sub2)

    for i in range(phe_hmm.n):

        state = phe_hmm.states[i]

        node_attributes = {}

        if phe_nodes_color:
            node_attributes['style'] = 'filled'
            node_attributes['fillcolor'] = node_colors[i]

        if phe_nodes_type:
            node_attributes['shape'] = 'ellipse'

        g_sub1.add_node(_pyd.Node(state, **node_attributes))

    for symbol in phe_hmm.symbols:

        node_attributes = {}

        if phe_nodes_color:
            node_attributes['style'] = 'filled'
            node_attributes['fillcolor'] = _default_color_symbol

        if phe_nodes_type:
            node_attributes['shape'] = 'hexagon'

        g_sub2.add_node(_pyd.Node(symbol, **node_attributes))

    for i in range(phe_hmm.n):

        state_i = phe_hmm.states[i]

        for j in range(phe_hmm.n):

            tp = phe_hmm.p[i, j]

            if tp > 0.0:

                state_j = phe_hmm.states[j]

                edge_attributes = {
                    'style': 'filled',
                    'color': _default_color_edge
                }

                if phe_edges_label:
                    edge_attributes['label'] = f' {round(tp, magnitude):.{magnitude}f} '
                    edge_attributes['fontsize'] = 9

                g.add_edge(_pyd.Edge(state_i, state_j, **edge_attributes))

        for j in range(phe_hmm.k):

            ep = phe_hmm.e[i, j]

            if ep > 0.0:

                symbol = phe_hmm.symbols[j]

                edge_attributes = {
                    'style': 'dashed',
                    'color': edge_colors[i] if phe_nodes_color else _default_color_edge
                }

                if phe_edges_label:
                    edge_attributes['label'] = f' {round(ep, magnitude):.{magnitude}f} '
                    edge_attributes['fontsize'] = 9

                g.add_edge(_pyd.Edge(state_i, symbol, **edge_attributes))

    f, a = _plot_graph_image(g, phe_dpi)

    return f, a


def _update_graph_bounds(ax, renderer):

    bounds = []

    for artist in [*ax.patches, *ax.texts, *ax.collections]:

        if not artist.get_visible():
            continue

        extent = artist.get_window_extent(renderer)
        values = _np.array(extent.extents)

        if _np.all(_np.isfinite(values)) and extent.width > 0.0 and extent.height > 0.0:
            bounds.append(values)

    if len(bounds) == 0:
        return False

    bounds = _np.array(bounds)
    padding = 8.0
    x_min = _np.min(bounds[:, 0]) - padding
    y_min = _np.min(bounds[:, 1]) - padding
    x_max = _np.max(bounds[:, 2]) + padding
    y_max = _np.max(bounds[:, 3]) + padding

    axes_bounds = ax.get_window_extent(renderer)

    if (x_min >= axes_bounds.x0 and y_min >= axes_bounds.y0 and
            x_max <= axes_bounds.x1 and y_max <= axes_bounds.y1):
        return False

    data_bounds = ax.transData.inverted().transform([[x_min, y_min], [x_max, y_max]])
    data_x_min, data_y_min = data_bounds[0]
    data_x_max, data_y_max = data_bounds[1]
    axis_x_min, axis_x_max = ax.get_xlim()
    axis_y_min, axis_y_max = ax.get_ylim()

    ax.set_xlim(min(axis_x_min, data_x_min), max(axis_x_max, data_x_max))
    ax.set_ylim(min(axis_y_min, data_y_min), max(axis_y_max, data_y_max))

    return True


def _on_graph_draw(event):

    axes = event.canvas.figure.axes

    if len(axes) != 1:
        return

    if _update_graph_bounds(axes[0], event.renderer):
        event.canvas.draw_idle()


def _fit_graph_bounds(ax):

    figure = ax.figure

    for _ in range(5):

        figure.canvas.draw()

        if not _update_graph_bounds(ax, figure.canvas.get_renderer()):
            break

    figure.canvas.mpl_connect('draw_event', _on_graph_draw)


def plot_hmm_standard(phs_hmm, phs_nodes_color, phs_nodes_shape, phe_edges_label, phs_dpi):

    g = phs_hmm.to_graph()
    positions = _nx.multipartite_layout(g, align='horizontal', subset_key='layer')
    magnitude = _calculate_magnitude(phs_hmm.p, phs_hmm.e)

    node_colors = _node_colors(phs_hmm.n) if phs_nodes_color else []
    edge_colors = _cp.deepcopy(node_colors) if phs_nodes_color else []

    f, a = _mplp.subplots(dpi=phs_dpi)

    for i, node in enumerate(g.nodes):

        if phs_nodes_color:
            if node in phs_hmm.states:
                node_color = node_colors[i]
            else:
                node_color = _default_color_symbol
        else:
            node_color = _default_color_node

        if phs_nodes_shape:
            if node in phs_hmm.states:
                node_shape = 'o'
            else:
                node_shape = 'H'
        else:
            node_shape = 'o'

        _nx.draw_networkx_nodes(g, positions, ax=a, nodelist=[node], node_color=node_color, node_shape=node_shape, node_size=_default_node_size, edgecolors=_color_black)

    _nx.draw_networkx_labels(g, positions, ax=a)

    edge_labels_curved, edge_labels_straight_state, edge_labels_straight_symbol = {}, {}, {}

    for i in range(phs_hmm.n):

        state_i = phs_hmm.states[i]

        for j in range(phs_hmm.n):

            tp = phs_hmm.p[i, j]

            if tp > 0.0:

                state_j = phs_hmm.states[j]

                edge = (state_i, state_j)
                edge_color = _default_color_edge

                if i != j and reversed(edge) in g.edges:

                    edge_length = abs(i - j)
                    edge_rad = 0.15 if edge_length == 1 else edge_length * 0.25
                    edge_connection = f'arc3, rad={edge_rad:f}'

                    if phe_edges_label:
                        edge_labels_curved[edge] = (edge_rad, f' {round(tp, magnitude):.{magnitude}f} ')

                else:

                    edge_connection = 'arc3'

                    if phe_edges_label:
                        edge_labels_straight_state[edge] = f' {round(tp, magnitude):.{magnitude}f} '

                _nx.draw_networkx_edges(g, positions, ax=a, edgelist=[edge], edge_color=edge_color, arrows=True, connectionstyle=edge_connection)

        for j in range(phs_hmm.k):

            ep = phs_hmm.e[i, j]

            if ep > 0.0:

                symbol_j = phs_hmm.symbols[j]

                edge = (state_i, symbol_j)
                edge_color = edge_colors[i] if phs_nodes_color else _default_color_edge

                if phe_edges_label:
                    edge_labels_straight_symbol[edge] = f' {round(ep, magnitude):.{magnitude}f} '

                _nx.draw_networkx_edges(g, positions, ax=a, edgelist=[edge], edge_color=edge_color, arrows=True, style='dashed')

    if len(edge_labels_straight_state) > 0:
        _nx.draw_networkx_edge_labels(g, positions, ax=a, edge_labels=edge_labels_straight_state)

    if len(edge_labels_straight_symbol) > 0:
        _nx.draw_networkx_edge_labels(g, positions, ax=a, edge_labels=edge_labels_straight_symbol, label_pos=0.25)

    if len(edge_labels_curved) > 0:
        _draw_edge_labels_curved(a, positions, edge_labels_curved)

    a.set_aspect('equal')
    _fit_graph_bounds(a)

    return f, a


def plot_mc_extended(pme_mc, pme_nodes_color, pme_nodes_shape, phe_edges_label, pme_dpi):

    magnitude = _calculate_magnitude(pme_mc.p)

    node_colors = _node_colors(len(pme_mc.communicating_classes)) if pme_nodes_color else []

    g = _pyd.Dot(graph_type='digraph')

    for i in range(pme_mc.size):

        state_i = pme_mc.states[i]

        node_attributes = {}

        if pme_nodes_color:
            for index, cc in enumerate(pme_mc.communicating_classes):
                if state_i in cc:
                    node_attributes['style'] = 'filled'
                    node_attributes['fillcolor'] = node_colors[index]
                    break

        if pme_nodes_shape:
            if state_i in pme_mc.transient_states:  # pragma: no cover
                node_attributes['shape'] = 'box'
            else:
                node_attributes['shape'] = 'ellipse'

        g.add_node(_pyd.Node(state_i, **node_attributes))

        for j in range(pme_mc.size):

            tp = pme_mc.p[i, j]

            if tp > 0.0:

                state_j = pme_mc.states[j]

                edge_attributes = {
                    'style': 'filled',
                    'color': _default_color_edge
                }

                if phe_edges_label:
                    edge_attributes['label'] = f' {round(tp, magnitude):.{magnitude}f} '
                    edge_attributes['fontsize'] = 9

                g.add_edge(_pyd.Edge(state_i, state_j, **edge_attributes))

    f, a = _plot_graph_image(g, pme_dpi)

    return f, a


def plot_mc_standard(pms_mc, pms_nodes_color, pms_nodes_shape, phe_edges_label, pms_dpi):

    g = pms_mc.to_graph()
    positions = _nx.spring_layout(g, seed=0)
    magnitude = _calculate_magnitude(pms_mc.p)

    node_colors = _node_colors(len(pms_mc.communicating_classes)) if pms_nodes_color else []

    f, a = _mplp.subplots(dpi=pms_dpi)

    for node in g.nodes:

        node_color = _default_color_node

        if pms_nodes_color:
            for index, cc in enumerate(pms_mc.communicating_classes):
                if node in cc:
                    node_color = node_colors[index]
                    break

        if pms_nodes_shape:
            if node in pms_mc.transient_states:  # pragma: no cover
                node_shape = 's'
            else:
                node_shape = 'o'
        else:
            node_shape = 'o'

        _nx.draw_networkx_nodes(g, positions, ax=a, nodelist=[node], node_color=node_color, node_shape=node_shape, node_size=_default_node_size, edgecolors=_color_black)

    _nx.draw_networkx_labels(g, positions, ax=a)

    edge_labels_curved, edge_labels_straight = {}, {}

    for i in range(pms_mc.size):

        state_i = pms_mc.states[i]

        for j in range(pms_mc.size):

            tp = pms_mc.p[i, j]

            if tp > 0.0:

                state_j = pms_mc.states[j]

                edge = (state_i, state_j)
                edge_color = _default_color_edge

                if i != j and reversed(edge) in g.edges:

                    edge_connection = 'arc3, rad=0.1'

                    if phe_edges_label:
                        edge_labels_curved[edge] = (0.1, f' {round(tp, magnitude):.{magnitude}f} ')

                else:

                    edge_connection = 'arc3'

                    if phe_edges_label:
                        edge_labels_straight[edge] = f' {round(tp, magnitude):.{magnitude}f} '

                _nx.draw_networkx_edges(g, positions, ax=a, edgelist=[edge], edge_color=edge_color, arrows=True, connectionstyle=edge_connection)

    if len(edge_labels_straight) > 0:
        _nx.draw_networkx_edge_labels(g, positions, ax=a, edge_labels=edge_labels_straight)

    if len(edge_labels_curved) > 0:
        _draw_edge_labels_curved(a, positions, edge_labels_curved)

    a.set_aspect('equal')
    _fit_graph_bounds(a)

    return f, a


def plot_sequence_heatmap(phm_walk_data, phm_dpi):

    walk_steps, walks = phm_walk_data
    plots_count = len(walks)

    f, a = _mplp.subplots(nrows=plots_count, constrained_layout=True, dpi=phm_dpi)
    a = [a] if plots_count == 1 else list(a.flat)

    color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 20)
    is_axes = []

    for a_current, (size, labels_name, labels, sequence) in zip(a, walks):

        sequence_matrix = _np.zeros((size, size), dtype=float)

        for i in range(1, walk_steps):
            sequence_matrix[sequence[i - 1], sequence[i]] += 1.0

        sequence_matrix /= _np.sum(sequence_matrix)

        a_current_is = a_current.imshow(sequence_matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)
        is_axes.append(a_current_is)

        _xticks_labels(a_current, size, labels_name, labels, True)
        _yticks_labels(a_current, size, labels_name, labels)

        a_current.grid(which='minor', color='k')

    color_map_ax, color_map_ax_kwargs = _mplcb.make_axes(a, drawedges=True, orientation='vertical', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    f.colorbar(is_axes[0], cax=color_map_ax, **color_map_ax_kwargs)

    f.suptitle('Sequence Plot (Heatmap)', fontsize=15.0, fontweight='bold')

    return f, a


def plot_sequence_histogram(ph_walk_data, ph_dpi):

    _, walks = ph_walk_data
    plots_count = len(walks)

    f, a = _mplp.subplots(nrows=plots_count, tight_layout=True, dpi=ph_dpi)
    a = [a] if plots_count == 1 else list(a.flat)

    for a_current, (size, labels_name, labels, sequence) in zip(a, walks):

        sequence_histogram = _np.bincount(sequence, minlength=size).astype(float)
        sequence_histogram /= _np.sum(sequence_histogram)

        a_current.bar(_np.arange(0.0, size, 1.0), sequence_histogram, edgecolor=_color_black, facecolor=_colors[0])

        _xticks_labels(a_current, size, labels_name, labels, False)
        _yticks_frequency(a_current, 0.0, 1.0)

    f.suptitle('Sequence Plot (Histogram)', fontsize=15.0, fontweight='bold')

    return f, a


def plot_sequence_matrix(pm_walk_data, pm_dpi):

    walk_steps, walks = pm_walk_data
    plots_count = len(walks)

    f, a = _mplp.subplots(nrows=plots_count, tight_layout=True, dpi=pm_dpi)
    a = [a] if plots_count == 1 else list(a.flat)

    color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 2)

    for a_current, (size, labels_name, labels, sequence) in zip(a, walks):

        sequence_matrix = _np.zeros((size, walk_steps), dtype=float)

        for index, state in enumerate(sequence):
            sequence_matrix[state, index] = 1.0

        a_current.imshow(sequence_matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        _xticks_steps(a_current, walk_steps)
        _yticks_labels(a_current, size, labels_name, labels)

        a_current.grid(which='minor', color='k')

    f.suptitle('Sequence Plot (Matrix)', fontsize=15.0, fontweight='bold')

    return f, a


def _generate_trellis_extended(gte_hmm, gte_initial_distribution, gte_symbols, gte_forward, gte_matrix, gte_states_path):

    n, f = gte_hmm.n, len(gte_symbols)

    g = _pyd.Dot(graph_type='digraph', compound='true', margin='0', splines='line')

    sub = _pyd.Subgraph('cluster_0', color='transparent')

    for row in range(n):

        sub_label = gte_hmm.states[row]
        sub.add_node(_pyd.Node(f'state{row}', color='transparent', label=sub_label, shape='plaintext'))

        if row > 0:
            sub.add_edge(_pyd.Edge(f'state{row - 1}', f'state{row}', style='invis'))

    g.add_subgraph(sub)

    for col in range(f):

        path_current = gte_states_path[col]

        sub_label = "<T<FONT POINT-SIZE='8'><SUB>0</SUB></FONT>>" if col == 0 else gte_hmm.symbols[gte_symbols[col]]
        sub = _pyd.Subgraph(f'cluster_{col + 1}', color='transparent', label=sub_label)

        for row in range(n):

            node_index = (row * f) + col
            node_color = _default_color_path if path_current == row else _default_color_node
            sub.add_node(_pyd.Node(f'node{node_index}', fillcolor=node_color, label=f'{round(gte_matrix[row, col], 2):.2f}', style='filled'))

            if row > 0:
                sub.add_edge(_pyd.Edge(f'node{((row - 1) * f) + col}', f'node{node_index}', style='invis'))

        g.add_subgraph(sub)

    for row in range(n):

        row_offset = row * f

        for col in range(f - 1):

            if (col == 0) and (not gte_initial_distribution[row] > 0.0):
                continue

            for row_next in range(n):

                if gte_hmm.p[row][row_next] > 0.0:

                    if gte_forward:
                        edge_from = f'node{row_offset + col}'
                        edge_to = f'node{(row_next * f) + col + 1}'
                    else:
                        edge_from = f'node{(row_next * f) + col + 1}'
                        edge_to = f'node{row_offset + col}'

                    edge_color = _default_color_path if gte_states_path[col] == row and gte_states_path[col + 1] == row_next else _color_black

                    g.add_edge(_pyd.Edge(edge_from, edge_to, color=edge_color, constraint='false'))

    return g


def _generate_trellis_standard(gts_hmm, gts_initial_distribution, gts_symbols, gts_forward, gts_matrix, gts_states_path):

    n, f = gts_hmm.n, len(gts_symbols)

    trellis = _nx.DiGraph()
    node_colors, node_edges, node_labels, node_positions = [], [], {}, {}

    node_index = 0

    for row in range(n):

        row_offset = float(n - row)

        for col in range(f):

            trellis.add_node(node_index)

            if gts_states_path[col] == row:
                node_colors.append(_default_color_path)
            else:
                node_colors.append(_default_color_node)

            node_edges.append(_color_black)
            node_labels[node_index] = f'{round(gts_matrix[row, col], 2):.2f}'
            node_positions[node_index] = (col + 1.0, row_offset)

            node_index += 1

    for row in range(n):

        row_offset = row * f

        for col in range(f - 1):

            if (col == 0) and (not gts_initial_distribution[row] > 0.0):
                continue

            for row_next in range(n):

                if gts_hmm.p[row][row_next] > 0.0:

                    if gts_forward:
                        edge = (row_offset + col, (row_next * f) + col + 1)
                    else:
                        edge = ((row_next * f) + col + 1, row_offset + col)

                    if (gts_states_path[col] == row) and (gts_states_path[col + 1] == row_next):
                        edge_color = _default_color_path
                    else:
                        edge_color = _color_black

                    trellis.add_edge(*edge, color=edge_color)

    for row in range(n):

        trellis.add_node(node_index)

        node_colors.append('none')
        node_edges.append('none')
        node_labels[node_index] = gts_hmm.states[row]
        node_positions[node_index] = (0.6, float(n - row))

        node_index += 1

    headers = [r"$\mathregular{T_0}$"] + [gts_hmm.symbols[symbol] for symbol in gts_symbols[1:]]

    for col, header in enumerate(headers):

        trellis.add_node(node_index)

        node_colors.append('none')
        node_edges.append('none')
        node_labels[node_index] = header
        node_positions[node_index] = (col + 1.0, n + 0.35)

        node_index += 1

    edge_colors = [trellis.edges[edge]['color'] for edge in trellis.edges()]

    return trellis, node_colors, node_edges, node_labels, node_positions, edge_colors


def plot_trellis_extended(ps_hmm, ps_initial_distribution, ps_symbols, ps_backward, ps_forward, ps_states_path, ps_dpi):

    f, a = _mplp.subplots(nrows=2, tight_layout=True, dpi=ps_dpi)
    a = list(a.flat)

    trellis = _generate_trellis_extended(ps_hmm, ps_initial_distribution, ps_symbols, False, ps_backward, ps_states_path)
    img, _, _, _, _ = _decode_image(trellis, ps_dpi)

    ax_current = a[0]
    ax_current.imshow(img)
    ax_current.axis('off')
    ax_current.set_title('Backward Trellis', fontsize=15.0, fontweight='normal', pad=1)

    trellis = _generate_trellis_extended(ps_hmm, ps_initial_distribution, ps_symbols, True, ps_forward, ps_states_path)
    img, _, _, _, _ = _decode_image(trellis, ps_dpi)

    ax_current = a[1]
    ax_current.imshow(img)
    ax_current.axis('off')
    ax_current.set_title('Forward Trellis', fontsize=15.0, fontweight='normal', pad=1)

    return f, a


def plot_trellis_standard(ps_hmm, ps_initial_distribution, ps_symbols, ps_backward, ps_forward, ps_states_path, ps_dpi):

    y_top = ps_hmm.n + 0.5

    f, a = _mplp.subplots(nrows=2, tight_layout=True, dpi=ps_dpi)
    a = list(a.flat)

    trellis, node_colors, node_edges, node_labels, node_positions, edge_colors = _generate_trellis_standard(ps_hmm, ps_initial_distribution, ps_symbols, False, ps_backward, ps_states_path)

    ax_current = a[0]
    _nx.draw_networkx(trellis, node_positions, ax=ax_current, edgecolors=node_edges, edge_color=edge_colors, font_size=9, labels=node_labels, node_color=node_colors, node_size=_default_node_size)
    ax_current.set_ylim(0.5, y_top)
    ax_current.axis('off')
    ax_current.set_title('Backward Trellis', fontsize=15.0, fontweight='normal', pad=1)

    trellis, node_colors, node_edges, node_labels, node_positions, edge_colors = _generate_trellis_standard(ps_hmm, ps_initial_distribution, ps_symbols, True, ps_forward, ps_states_path)

    ax_current = a[1]
    _nx.draw_networkx(trellis, node_positions, ax=ax_current, edgecolors=node_edges, edge_color=edge_colors, font_size=9, labels=node_labels, node_color=node_colors, node_size=_default_node_size)
    ax_current.set_ylim(0.5, y_top)
    ax_current.axis('off')
    ax_current.set_title('Forward Trellis', fontsize=15.0, fontweight='normal', pad=1)

    return f, a
