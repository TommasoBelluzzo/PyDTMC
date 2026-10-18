Reduction & Aggregation
=======================

| This guide explores how a Markov chain can be simplified by combining states with similar behavior.
| The example follows a machine operating in four conditions, which we want to describe using only two broader conditions.

Model Construction
------------------

| The machine has four states: ``Normal A``, ``Normal B``, ``Degraded A``, and ``Degraded B``.
| The two normal states represent different operating modes.
| The two degraded states represent different types of reduced performance.

The transition matrix is:

.. math::

   P = \begin{pmatrix}
      0.50 & 0.30 & 0.10 & 0.10 \\
      0.20 & 0.60 & 0.15 & 0.05 \\
      0.20 & 0.10 & 0.40 & 0.30 \\
      0.10 & 0.20 & 0.20 & 0.50
   \end{pmatrix}

| For example, from ``Normal A`` the machine has an 80% chance of remaining in a normal condition and a 20% chance of becoming degraded.
| These combined probabilities are the same when starting from ``Normal B``.

Using PyDTMC, the chain can be created as follows:

.. code-block:: python

   from pydtmc import MarkovChain

   p = [
       [0.50, 0.30, 0.10, 0.10],
       [0.20, 0.60, 0.15, 0.05],
       [0.20, 0.10, 0.40, 0.30],
       [0.10, 0.20, 0.20, 0.50]
   ]

   states = ['Normal A', 'Normal B', 'Degraded A', 'Degraded B']

   mc = MarkovChain(p, states)

Grouping States
---------------

| A **partition** divides the states into nonoverlapping groups, with every state belonging to exactly one group.
| Here, we combine ``Normal A`` and ``Normal B`` into ``Normal``.
| We also combine ``Degraded A`` and ``Degraded B`` into ``Degraded``.

The partition is:

.. math::

   \mathcal{P} = \{\{\text{Normal A},\text{Normal B}\}, \{\text{Degraded A},\text{Degraded B}\}\}

| The goal is to follow whether the machine is normal or degraded without tracking its specific operating mode.
| However, combining states does not always produce another valid Markov chain.

Lumpability
-----------

| A partition is **lumpable** when all states in the same group have identical probabilities of moving to each destination group.
| This ensures that the next group's probabilities depend only on the current group, not on the particular state inside it.

| Both normal states have probability ``0.80`` of moving to ``Normal`` and ``0.20`` of moving to ``Degraded``.
| Both degraded states have probability ``0.30`` of moving to ``Normal`` and ``0.70`` of remaining in ``Degraded``.
| The partition therefore satisfies the lumpability condition.

The reduced transition matrix is:

.. math::

   \widetilde{P} = \begin{pmatrix}
      0.8 & 0.2 \\
      0.3 & 0.7
   \end{pmatrix}

Its rows and columns represent ``Normal`` and ``Degraded``, in that order.

Lumping
-------

The ``lump`` method constructs the reduced Markov chain from the partition:

.. code-block:: python

   partition = [
       ['Normal A', 'Normal B'],
       ['Degraded A', 'Degraded B']
   ]

   reduced = mc.lump(partition)

   print(reduced.states)
   print(reduced.p)

| The resulting chain has two states, one for each group in the partition.
| Its transition probabilities agree with the reduced matrix above.
| Unlike an approximation, valid lumping preserves the exact probabilities of transitions between groups.

Long-Term Behavior
------------------

| The reduced chain also describes the long-run proportions of time spent in normal and degraded conditions.
| Its stationary distribution is:

.. code-block:: python

   print(reduced.pi)

.. code-block:: text

   [0.6, 0.4]

| In the long run, the machine is normal 60% of the time and degraded 40% of the time.
| These probabilities also equal the sums of the original chain's stationary probabilities within each group.

Spectral Aggregation
--------------------

| Exact lumping requires states within each group to have matching transition probabilities toward every group.
| When this condition does not hold, **spectral aggregation** can instead construct a smaller approximate chain.
| It uses the original transition structure to identify a reduced model without specifying the groups beforehand.

PyDTMC can construct a two-state spectral aggregation as follows:

.. code-block:: python

   approximate = mc.spectral_aggregation(2)

   print(approximate.states)
   print(approximate.p)

| Unlike ``lump``, spectral aggregation does not guarantee that the reduced chain reproduces every group-level transition probability exactly.
| It is useful when a simpler description is desired even though exact lumping is unavailable or unsuitable.