Visualization
=============

| Visualizations help us inspect transitions, follow changing probabilities, and interpret simulated behavior.
| This guide introduces several ways to visualize Markov chains and hidden Markov models using PyDTMC.

Model Construction
------------------

| We begin with a weather model containing three states: ``Sunny``, ``Cloudy``, and ``Rainy``.
| The transition matrix describes the probability of tomorrow's weather given today's weather.

.. math::

   P = \begin{pmatrix}
      0.8 & 0.2 & 0.0 \\
      0.3 & 0.4 & 0.3 \\
      0.0 & 0.6 & 0.4
   \end{pmatrix}

Using PyDTMC, the model can be created as follows:

.. code-block:: python

   from pydtmc import MarkovChain

   p = [
       [0.8, 0.2, 0.0],
       [0.3, 0.4, 0.3],
       [0.0, 0.6, 0.4]
   ]

   mc = MarkovChain(p, ['Sunny', 'Cloudy', 'Rainy'])

Plotting functions should also be imported:

.. code-block:: python

   from pydtmc.plotting import (
       plot_convergence,
       plot_graph,
       plot_matrix,
       plot_redistributions,
       plot_sequence
   )

Graphs
------

| A **graph** represents each state as a node and each possible transition as an arrow.
| This makes it easier to see which states can be reached directly.
| For example, ``Sunny`` and ``Rainy`` have no direct transition between them.

.. code-block:: python

   plot_graph(mc)

| The graph shows the connections among the weather states.
| Arrow labels indicate transition probabilities, including the probabilities of remaining in the same state.

Transition Matrix
-----------------

| A **matrix heatmap** represents transition probabilities using colors.
| Each row describes the possible next states from one starting state.
| Stronger colors indicate higher probabilities.

.. code-block:: python

   plot_matrix(mc)

| The ``Sunny``-to-``Sunny`` entry has probability ``0.8``, while the ``Sunny``-to-``Rainy`` entry is zero.
| The heatmap makes these differences easy to compare.

Distribution Evolution
----------------------

| A state distribution describes the probabilities of being in each weather condition.
| The ``plot_redistributions`` function shows how these probabilities change over successive transitions.

.. code-block:: python

   plot_redistributions(mc, 10, initial_status='Sunny')

| The plot begins with probability ``1.0`` for ``Sunny``.
| Over time, the three probabilities approach the stationary distribution ``[0.5000, 0.3333, 0.1667]``.

| The same information can be displayed as a heatmap:

.. code-block:: python

   plot_redistributions(mc, 10, initial_status='Sunny', plot_type='heatmap')

| Here, each column represents a step and each row represents a weather state.
| The colors show how the probability assigned to each state changes over time.

Convergence
-----------

| The **total variation distance** measures how far a state distribution is from the stationary distribution.
| A distance of zero means the distributions are identical.

.. code-block:: python

   plot_convergence(mc, 10, initial_status='Sunny')

| The distance decreases as the initial certainty about sunny weather loses its influence.
| This plot summarizes convergence in a single curve rather than showing three separate state probabilities.

Simulated Sequences
-------------------

| A simulation produces one random sequence of weather conditions.
| A histogram of that sequence shows how frequently each state was visited.

.. code-block:: python

   plot_sequence(mc, 1000, initial_state='Sunny', seed=42)

| The histogram displays the observed frequencies, not the exact stationary probabilities.
| In a sufficiently long simulation, these frequencies tend to approach the stationary distribution.

Hidden Markov Models
--------------------

| Hidden Markov models also include an emission matrix, which connects hidden states to observable symbols.
| We can visualize these probabilities separately from transitions between hidden states.

.. code-block:: python

   from pydtmc import HiddenMarkovModel
   from pydtmc.plotting import plot_trellis

   hmm = HiddenMarkovModel(
       [[0.8, 0.2], [0.3, 0.7]],
       [[0.9, 0.1], [0.2, 0.8]],
       ['Dry', 'Wet'],
       ['No Umbrella', 'Umbrella']
   )

   plot_matrix(hmm, underlying_matrix='emission')

| Rows correspond to hidden weather conditions and columns to observed umbrella symbols.
| The heatmap shows, for example, that ``Umbrella`` is more likely under ``Wet`` than ``Dry``.

| A **trellis diagram** illustrates how hidden-state sequences are evaluated against observations.
| PyDTMC highlights the most probable hidden-state path found by the Viterbi algorithm.

.. code-block:: python

   plot_trellis(hmm, 5, initial_state='Dry', seed=42, force_standard=True)