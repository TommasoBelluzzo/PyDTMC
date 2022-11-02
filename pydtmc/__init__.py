# -*- coding: utf-8 -*-

__title__ = 'PyDTMC'
__version__ = '7.0.0'
__author__ = 'Tommaso Belluzzo'

__all__ = [
    'ValidationError',
    'MarkovChain', 'HiddenMarkovModel',
    'assess_first_order', 'assess_homogeneity', 'assess_markov_property', 'assess_stationarity', 'assess_theoretical_compatibility',
    'plot_comparison', 'plot_eigenvalues', 'plot_graph', 'plot_redistributions', 'plot_walk'
]

from pydtmc.exceptions import (
    ValidationError
)

from pydtmc.markov_chain import (
    MarkovChain
)

from pydtmc.hidden_markov_model import (
    HiddenMarkovModel
)

from pydtmc.assessments import (
    assess_first_order,
    assess_homogeneity,
    assess_markov_property,
    assess_stationarity,
    assess_theoretical_compatibility
)

from pydtmc.plotting import (
    plot_comparison,
    plot_eigenvalues,
    plot_graph,
    plot_redistributions,
    plot_walk
)

if __name__ == '__main__':

    from matplotlib.pyplot import (
        interactive as _mplp_interactive,
        isinteractive as _mplp_isinteractive,
        figure as _mplp_figure,
        Rectangle as _mplp_Rectangle,
        show as _mplp_show,
        subplots as _mplp_subplots,
        subplots_adjust as _mplp_subplots_adjust
    )

    mc = MarkovChain([[0.4, 0.6], [0.7, 0.3]])
    #f, a = plot_graph(mc)
    #_mplp_show(block=False)

    z = 1

    hmm = HiddenMarkovModel([[0.4, 0.6], [0.7, 0.3]], [[0.1, 0.5, 0.4], [0.2, 0.6, 0.2]])
    f, a = plot_graph(hmm)
    _mplp_show(block=False)

    z = 1