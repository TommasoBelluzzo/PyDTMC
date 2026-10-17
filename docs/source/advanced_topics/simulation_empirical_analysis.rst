Simulation & Empirical Analysis
===============================

| A Markov chain describes how a process moves between states using transition probabilities.
| Simulation produces one possible sequence of states according to these probabilities.
| By examining the sequence, we can estimate the behavior of the process from observations.

Model Construction
------------------

Consider the three-state weather model introduced in the :doc:`Markov Chain Analysis <markov_chain_analysis>` guide:

.. math::

   P = \begin{pmatrix}
      0.8 & 0.2 & 0.0 \\
      0.3 & 0.4 & 0.3 \\
      0.0 & 0.6 & 0.4
   \end{pmatrix}

| The states are ``Sunny``, ``Cloudy``, and ``Rainy``.
| Each row contains the probabilities of tomorrow's weather given today's weather.

.. code-block:: python

   from pydtmc import MarkovChain

   p = [
       [0.8, 0.2, 0.0],
       [0.3, 0.4, 0.3],
       [0.0, 0.6, 0.4]
   ]

   states = ['Sunny', 'Cloudy', 'Rainy']

   mc = MarkovChain(p, states)

Generating Observations
-----------------------

| A simulation starts in one state and randomly selects each subsequent state.
| The choices follow the transition probabilities of the current state.

.. code-block:: python

   sequence = mc.simulate(2000, initial_state='Sunny', seed=42)

   print(sequence[:10])
   print(len(sequence))

| The sequence contains ``2001`` states: the initial state and one state for each of the ``2000`` transitions.
| It represents one possible weather history, not the average behavior of all possible histories.
| A fixed random seed makes the result reproducible.

Empirical State Frequencies
---------------------------

| We can count how often each state appears in the simulated sequence.
| Dividing these counts by the sequence length gives the **empirical state frequencies**.
| These describe the observed share of time spent in each weather condition.

.. code-block:: python

   import numpy as np

   frequencies = np.array([sequence.count(state) for state in states]) / len(sequence)

   print(np.round(frequencies, 4))
   print(np.round(mc.pi[0], 4))

| The first vector contains the observed frequencies; the second is the theoretical stationary distribution.
| For this **ergodic** chain, empirical frequencies approach :math:`(1/2, 1/3, 1/6)` as the sequence grows.
| A finite simulation will generally produce slightly different proportions.

Estimating Transition Probabilities
-----------------------------------

| A state frequency tells us how often a state was observed, but not how the process moves between states.
| To estimate transition probabilities, we count consecutive pairs of states.
| For example, every occurrence of ``Sunny`` followed by ``Cloudy`` counts as one sunny-to-cloudy transition.

| Let :math:`c_{ij}` be the number of observed transitions from state :math:`i` to state :math:`j`.
| The **maximum-likelihood estimate** of the transition probability is:

.. math::

   \widehat{P}_{ij} = \frac{c_{ij}}{\sum_k c_{ik}}

| Each transition count is divided by the total number of transitions leaving its origin state.
| The resulting matrix describes the transitions observed in the data.

PyDTMC can perform this estimation directly:

.. code-block:: python

   estimated = MarkovChain.fit_sequence('mle', states, sequence)
   print(np.round(estimated.p, 4))

| The fitted matrix should resemble the original transition matrix.
| Its entries generally differ because the simulation contains a finite number of random transitions.

Comparing Model & Observations
------------------------------

| We can compare the original probabilities with those estimated from the simulated sequence.
| The differences show how closely this particular sample reproduces the model.

.. code-block:: python

   print(np.round(mc.p, 4))
   print(np.round(estimated.p, 4))
   print(np.round(np.abs(mc.p - estimated.p), 4))

| The last matrix shows the absolute difference for each transition probability.
| Smaller differences indicate closer agreement between the simulated observations and the original model.
| With more observations, the estimated probabilities tend to approach the original values of the chain.

Simulation & Probability
------------------------

| Simulation and probability calculations answer different questions.
| The ``simulate`` method generates a single random history.
| The ``redistribute`` method calculates the probability of each state after a specified number of transitions.

.. code-block:: python

   print(sequence[:11])
   print(np.round(mc.redistribute(10, 'Sunny'), 4))

| The first result shows the states visited in one simulation during ten transitions.
| The second gives the state probabilities after ten transitions, starting from ``Sunny``.
| An individual simulated trajectory does not need to follow the most likely outcome at every step.