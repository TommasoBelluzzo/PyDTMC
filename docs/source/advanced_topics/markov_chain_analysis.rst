Markov Chain Analysis
=====================

| This guide explores the structure and long-term behavior of a Markov chain using a simple weather model.
| The model has three states: ``Sunny``, ``Cloudy``, and ``Rainy``.

Model Construction
------------------

Suppose the weather changes each day according to the following transition matrix:

.. math::

   P = \begin{pmatrix}
      0.8 & 0.2 & 0.0 \\
      0.3 & 0.4 & 0.3 \\
      0.0 & 0.6 & 0.4
   \end{pmatrix}

| Each row represents today's weather, while each column represents tomorrow's weather.
| For example, a sunny day is followed by another sunny day with probability ``0.8`` and by a cloudy day with probability ``0.2``.
| Direct transitions between sunny and rainy conditions have probability zero.

Using PyDTMC, the chain can be created as follows:

.. code-block:: python

   from pydtmc import MarkovChain

   p = [
       [0.8, 0.2, 0.0],
       [0.3, 0.4, 0.3],
       [0.0, 0.6, 0.4]
   ]

   mc = MarkovChain(p, ['Sunny', 'Cloudy', 'Rainy'])

Communicating Classes
---------------------

| A state is reachable from another if the chain can move between them in one or more steps.
| Two states **communicate** when each is reachable from the other.
| A **communicating class** is a group of states that all communicate with one another.

| In this model, sunny and rainy conditions cannot follow each other directly, but both can be reached through the cloudy state.
| All three states therefore belong to one communicating class. A chain with only one communicating class is called **irreducible**.

| A communicating class is **recurrent** when, after entering it, the chain cannot leave it.
| Here, the class contains every possible weather condition, so the chain always remains within it.
| In this finite model, this also means that each state is eventually revisited with probability 1.

These properties can be inspected as follows:

.. code-block:: python

   print(mc.communicating_classes)
   print(mc.recurrent_classes)
   print(mc.is_irreducible)

Periodicity and Ergodicity
--------------------------

| Some chains follow a fixed cycle. For example, a chain may alternate between two states.
| In this case, it can return to either state only after an even number of steps.
| The **period** describes this pattern of possible return times.
| A chain is **aperiodic** when its returns are not restricted to such a repeating cycle; its period is ``1``.

| In the weather model, each state can remain unchanged for one day.
| A return after just one step is therefore possible, so every state has period ``1`` and the chain is aperiodic.

| A finite chain that is both irreducible and aperiodic is called **ergodic**.
| In practical terms, its long-term state probabilities settle to the same values regardless of the starting state.

.. code-block:: python

   print(mc.period)
   print(mc.is_aperiodic)
   print(mc.is_ergodic)

Stationary Distribution
-----------------------

| At any given time, we can describe the weather by assigning a probability to each of the three states.
| These probabilities form a **state distribution**. For example, if today is certainly sunny, the distribution is ``[1.0, 0.0, 0.0]``.
| After one day, the distribution becomes ``[0.8, 0.2, 0.0]`` because there is an 80% chance of sun and a 20% chance of clouds.

| As the chain continues to evolve, these probabilities change.
| For an ergodic chain, they eventually approach a fixed set of probabilities, independent of the initial state.
| This is the **stationary distribution**: these probabilities remain constant over time, even as the weather changes.

Mathematically, a stationary distribution is a probability vector :math:`\pi` satisfying:

.. math::

   \pi P = \pi, \qquad \sum_i \pi_i = 1

Using PyDTMC, it can be computed as follows:

.. code-block:: python

   print(mc.pi[0])

For this chain, the unique stationary distribution is:

.. math::

   \pi = \left(\frac{1}{2},\frac{1}{3},\frac{1}{6}\right)

| In the long run, the probability of sunny weather approaches 50%, cloudy weather about 33.33%, and rainy weather about 16.67%.
| Over a sufficiently long observation period, these are also the proportions of days spent in each state.

Distribution Evolution
----------------------

| The ``propagate`` method computes how state probabilities change after a given number of transitions.
| Suppose today is certainly sunny, so the initial distribution is ``[1.0, 0.0, 0.0]``.

.. code-block:: python

   import numpy as np

   print(np.round(mc.propagate(2, 'Sunny'), 4))
   print(np.round(mc.propagate(10, 'Sunny'), 4))

The resulting distributions are:

.. code-block:: text

   [0.7000, 0.2400, 0.0600]
   [0.5072, 0.3300, 0.1628]

| After two transitions, there is a 70% chance of sun, a 24% chance of clouds, and a 6% chance of rain.
| After ten transitions, these probabilities are already close to the stationary distribution ``[0.5000, 0.3333, 0.1667]``.
| As more transitions are added, the probabilities converge to the stationary values described above.

Simulation
----------

A simulation generates one possible sequence of weather conditions:

.. code-block:: python

   print(mc.simulate(10, initial_state='Sunny', seed=42))

| Unlike ``propagate``, which calculates probabilities, ``simulate`` produces one random sequence of states.
| Over a very long simulation, the proportions of time spent in each state approach those of the stationary distribution.