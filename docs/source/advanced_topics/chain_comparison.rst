Chain Comparison
================

| This guide compares two Markov chains that describe the same process under different conditions.
| The example follows a machine that can be ``Operational`` or ``Repair``.

Model Construction
------------------

| Suppose two maintenance policies affect how often the machine changes state.
| Under the first policy, an operational machine remains operational with probability ``0.90``.
| Under the second policy, that probability falls to ``0.80``, but repairs finish more quickly.

The transition matrices are:

.. math::

   P = \begin{pmatrix}
      0.90 & 0.10 \\
      0.20 & 0.80
   \end{pmatrix},
   \qquad
   Q = \begin{pmatrix}
      0.80 & 0.20 \\
      0.30 & 0.70
   \end{pmatrix}

| Rows and columns follow the order ``Operational`` and ``Repair``.
| The first matrix describes the current policy, while the second describes an alternative.

Using PyDTMC, both chains can be created as follows:

.. code-block:: python

   from pydtmc import MarkovChain

   states = ['Operational', 'Repair']

   current = MarkovChain([
       [0.90, 0.10],
       [0.20, 0.80]
   ], states)

   alternative = MarkovChain([
       [0.80, 0.20],
       [0.30, 0.70]
   ], states)

Transition Probabilities
------------------------

| The two chains describe the same states but assign different probabilities to their transitions.
| Comparing corresponding entries shows how the maintenance policies differ.

.. code-block:: python

   import numpy as np

   print(np.round(alternative.p - current.p, 2))

The difference matrix is:

.. code-block:: text

   [[-0.1,  0.1]
    [ 0.1, -0.1]]

| Under the alternative policy, the chance of entering ``Repair`` from ``Operational`` is ten percentage points higher.
| The chance of returning to ``Operational`` from ``Repair`` is also ten percentage points higher.
| These differences describe individual transitions, not the long-term proportion of time in each state.

Stationary Distributions
------------------------

| The **stationary distribution** describes the long-run proportion of time spent in each state.
| Comparing stationary distributions reveals how the policies affect machine availability.

.. code-block:: python

   print(np.round(current.pi[0], 4))
   print(np.round(alternative.pi[0], 4))

The resulting distributions are:

.. code-block:: text

   [0.6667, 0.3333]
   [0.6000, 0.4000]

| With the current policy, the machine is operational approximately 66.67% of the time in the long run.
| With the alternative policy, that proportion falls to 60%.
| Faster repairs do not fully compensate for the more frequent breakdowns.

Entropy Rates
-------------

| The **entropy rate** measures the average uncertainty in the next state when the chain is in its stationary regime.
| A higher entropy rate indicates less predictable transitions on average.

.. code-block:: python

   print(round(current.entropy_rate, 4))
   print(round(alternative.entropy_rate, 4))

The entropy rates, in nats per transition, are:

.. code-block:: text

   0.3835
   0.5446

| The alternative policy has a higher entropy rate, so its state transitions are less predictable on average.
| This measure describes uncertainty, not whether a maintenance policy is preferable.

Merging Chains
--------------

| Suppose we want to explore a maintenance policy between the current and alternative policies.
| Rather than defining new transition probabilities manually, we can combine those of the two existing models.
| The ``merge_with`` method uses a weight between ``0`` and ``1`` to control their contributions.

For a weight of ``0.5``, both policies contribute equally:

.. math::

   \widetilde{P} = 0.5P + 0.5Q
   = \begin{pmatrix}
      0.85 & 0.15 \\
      0.25 & 0.75
   \end{pmatrix}

.. code-block:: python

   merged = current.merge_with(alternative, 0.5)
   print(merged.p)

.. code-block:: text

   [[0.85, 0.15]
    [0.25, 0.75]]

| Each transition probability is the average of the corresponding probabilities in the two original chains.
| The stationary distribution of the merged chain is obtained from its new transition matrix:

.. code-block:: python

   print(np.round(merged.pi[0], 4))

.. code-block:: text

   [0.625, 0.375]

| In the long run, the merged policy keeps the machine ``Operational`` 62.5% of the time and under ``Repair`` 37.5% of the time.
| This lies between the long-run operational probabilities of the two original policies: 66.67% and 60%.
| Although the transition matrices are averaged, their stationary distributions are not generally averaged.