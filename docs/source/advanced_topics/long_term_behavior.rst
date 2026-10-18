Long-Term Behavior
==================

| This guide explores the long-term behavior of a Markov chain through a machine maintenance model.
| The machine can be either ``Operational`` or under ``Repair``.
| We will study how often each state occurs, how long it takes to return, and the rewards earned over time.

Model Construction
------------------

Suppose the machine changes state each day according to the following transition matrix:

.. math::

   P = \begin{pmatrix}
      0.9 & 0.1 \\
      0.4 & 0.6
   \end{pmatrix}

| An operational machine remains operational the next day with probability ``0.9``.
| A machine under repair becomes operational the next day with probability ``0.4``.

Using PyDTMC, the chain can be created as follows:

.. code-block:: python

   from pydtmc import MarkovChain

   p = [[0.9, 0.1], [0.4, 0.6]]
   mc = MarkovChain(p, ['Operational', 'Repair'])

Stationary Distribution
-----------------------

| The machine may change state each day, but its state probabilities can eventually settle.
| The **stationary distribution** describes these stable probabilities.
| If the machine starts with this distribution, the probabilities stay unchanged after every transition.

| Both states can reach each other, and both have positive self-transition probabilities.
| The chain is therefore **ergodic**, so its state probabilities converge to a unique stationary distribution.

.. code-block:: python

   print(mc.is_ergodic)
   print(mc.pi[0])

For this chain, the stationary distribution is:

.. math::

   \pi = (0.8, 0.2)

Over the long run, the machine is operational 80% of the time and under repair 20% of the time.

Distribution Evolution
----------------------

| Suppose the machine is operational today.
| As days pass, the probability of each state approaches its stationary value.

.. code-block:: python

   import numpy as np

   print(np.round(mc.redistribute(2, 'Operational'), 4))
   print(np.round(mc.redistribute(10, 'Operational'), 4))

The resulting distributions are:

.. code-block:: text

   [0.8500, 0.1500]
   [0.8002, 0.1998]

| After two transitions, the probability of being operational is 85%.
| After ten transitions, it is already close to the stationary value of 80%.

Recurrence Times
----------------

| A **return time** is the number of steps until a state is visited again, starting from that state.
| The **mean recurrence time** is the expected number of steps until the first return.
| Remaining in the same state for one step counts as a return.

For a finite irreducible chain, the mean recurrence time of state :math:`i` is:

.. math::

   m_i = \frac{1}{\pi_i}

It can be computed with PyDTMC:

.. code-block:: python

   print(mc.recurrence_times('mean'))

The mean recurrence times are:

.. code-block:: text

   [1.25, 5.00]

| The expected return time is 1.25 days for ``Operational`` and 5 days for ``Repair``.
| The more frequently a state is visited in the long run, the shorter its mean recurrence time.

Long-Term Rewards
-----------------

| A **state reward** assigns a numerical value to each state.
| Suppose an operational day earns ``10`` units, while a day under repair costs ``5`` units.
| We represent these rewards by the vector :math:`r = (10, -5)`.

The expected reward per day in the stationary regime is:

.. math::

   \pi r = 0.8 \cdot 10 + 0.2 \cdot (-5) = 7

| In the long run, the average reward approaches 7 units per day.
| The reward of an individual day still depends on the machine's actual state.

PyDTMC can also calculate the expected cumulative reward over a finite number of transitions:

.. code-block:: python

   rewards = [10, -5]
   print(np.round(mc.expected_rewards(10, rewards), 4))

The resulting values are:

.. code-block:: text

   [82.9971, 53.0117]

| Starting from ``Operational``, the expected cumulative reward is about 83 units.
| Starting from ``Repair``, it is about 53 units.
| These totals include the initial day's reward and the rewards after each of the next ten transitions.
| Over longer periods, the average expected reward per day approaches 7 units from either starting state.