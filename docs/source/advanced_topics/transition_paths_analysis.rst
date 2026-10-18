Transition Path Analysis
========================

| This guide explores how a Markov chain moves between two conditions and which intermediate states contribute to those transitions.
| The example follows a system whose temperature can be ``Cool``, ``Warm``, or ``Hot``.

| We want to understand transitions from ``Cool`` to ``Hot``.
| A **reactive trajectory** starts when the system last leaves ``Cool`` and ends when it reaches ``Hot`` without returning to ``Cool`` first.
| Transition path analysis describes the probabilities and probability flow associated with such trajectories.

Model Construction
------------------

| The system can remain at its current temperature or move to a neighboring temperature level.
| A direct transition between ``Cool`` and ``Hot`` is not possible.
| Unlike an absorbing process, the system can also cool down after becoming hot.

The transition matrix is:

.. math::

   P = \begin{pmatrix}
      0.7 & 0.3 & 0.0 \\
      0.2 & 0.6 & 0.2 \\
      0.0 & 0.3 & 0.7
   \end{pmatrix}

Using PyDTMC, the chain can be created as follows:

.. code-block:: python

   from pydtmc import MarkovChain

   p = [
       [0.7, 0.3, 0.0],
       [0.2, 0.6, 0.2],
       [0.0, 0.3, 0.7]
   ]

   mc = MarkovChain(p, ['Cool', 'Warm', 'Hot'])

Forward Committor Probabilities
-------------------------------

| The **forward committor** is the probability of reaching ``Hot`` before returning to ``Cool``, starting from a given state.
| Its value is ``0`` at ``Cool`` and ``1`` at ``Hot``.
| At ``Warm``, the system may move toward either endpoint, so the outcome is uncertain.

.. code-block:: python

   print(mc.committor_probabilities('forward', ['Cool'], ['Hot']))

The resulting probabilities are:

.. code-block:: text

   [0.0, 0.5, 1.0]

| Starting from ``Warm``, there is a 50% chance of reaching ``Hot`` before ``Cool``.
| Remaining at ``Warm`` for one or more steps does not change which endpoint is reached first.

Backward Committor Probabilities
--------------------------------

| The **backward committor** looks at the trajectory in reverse.
| It gives the probability that the chain last visited ``Cool`` rather than ``Hot``, assuming stationarity.

.. code-block:: python

   print(mc.committor_probabilities('backward', ['Cool'], ['Hot']))

The resulting probabilities are:

.. code-block:: text

   [1.0, 0.5, 0.0]

| At ``Warm``, the probability that the system last visited ``Cool`` rather than ``Hot`` is ``0.5``.
| The forward and backward probabilities both equal ``0.5`` at ``Warm`` in this example, but describe different questions.

Reactive Density
----------------

| The **reactive density** measures how frequently a state is visited during transitions from ``Cool`` to ``Hot``.
| It considers only transitions that reach ``Hot`` before returning to ``Cool``.
| It combines the stationary probability with the forward and backward committors.

For a state :math:`i`, it is defined as:

.. math::

   \eta_i = \pi_i q_i^- q_i^+

.. code-block:: python

   import numpy as np

   print(np.round(mc.reactive_density(['Cool'], ['Hot']), 4))

The resulting values are approximately:

.. code-block:: text

   [0.0000, 0.1071, 0.0000]

| Only ``Warm`` has a positive reactive density, because every transition from ``Cool`` to ``Hot`` must pass through it.
| The value ``0.1071`` means that about 10.71% of stationary observations are at ``Warm`` along an ongoing reactive trajectory.
| The endpoints have zero reactive density because they mark the beginning and end of those trajectories.

Reactive Flux
-------------

| The **reactive flux** measures stationary probability flow along transitions that contribute to reaching ``Hot`` from ``Cool``.
| The **net reactive flux** removes opposing flows, showing the direction of the remaining progress toward the target.

.. code-block:: python

   print(np.round(mc.reactive_flux(True, ['Cool'], ['Hot']), 4))

The resulting matrix is:

.. code-block:: text

   [[0.0000, 0.0429, 0.0000]
    [0.0000, 0.0000, 0.0429]
    [0.0000, 0.0000, 0.0000]]

| Rows identify the state being left and columns identify the state being entered, in the order ``Cool``, ``Warm``, and ``Hot``.
| The value ``0.0429`` at row ``Cool`` and column ``Warm`` represents net reactive flow along that transition.
| The same flow continues from ``Warm`` to ``Hot``.
| In this model, the route ``Cool`` → ``Warm`` → ``Hot`` is the only way to reach the target without revisiting ``Cool``.