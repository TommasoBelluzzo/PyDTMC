Hitting & Passage Problems
==========================

| This guide explores how long a Markov chain takes to reach particular states.
| The example models network traffic with three levels: ``Low``, ``Moderate``, and ``High``.

Model Construction
------------------

Suppose network traffic changes at regular intervals according to this transition matrix:

.. math::

   P = \begin{pmatrix}
      0.50 & 0.50 & 0.00 \\
      0.25 & 0.50 & 0.25 \\
      0.00 & 0.50 & 0.50
   \end{pmatrix}

| Each row represents the current traffic level, and each column represents the next level.
| Traffic can remain unchanged or move to a neighboring level.
| A direct transition between ``Low`` and ``High`` is not possible.

Using PyDTMC, the chain can be created as follows:

.. code-block:: python

   from pydtmc import MarkovChain

   p = [
       [0.50, 0.50, 0.00],
       [0.25, 0.50, 0.25],
       [0.00, 0.50, 0.50]
   ]

   mc = MarkovChain(p, ['Low', 'Moderate', 'High'])

Hitting Probabilities
---------------------

| A **hitting event** occurs when the chain reaches a chosen state for the first time.
| The **hitting probability** is the chance of eventually reaching that state.
| For example, we may ask whether traffic will ever become ``High``.

PyDTMC computes this probability from every possible starting state:

.. code-block:: python

   print(mc.hitting_probabilities(['High']))

The result is:

.. code-block:: text

   [1.0, 1.0, 1.0]

| Starting from either ``Low`` or ``Moderate``, traffic eventually reaches ``High`` with probability 1.
| The probability is also 1 when starting at ``High``, because the target has already been reached.
| This does not mean that traffic will remain ``High`` afterward.

Hitting Times
-------------

| The **hitting time** is the number of steps needed to reach a target state for the first time.
| The **mean hitting time** is the expected number of steps until that happens.
| If the chain starts in the target state, its hitting time is zero.

For the ``High`` traffic level, the expected hitting times can be computed as follows:

.. code-block:: python

   print(mc.hitting_times('mean', ['High']))

The result is:

.. code-block:: text

   [8.0, 6.0, 0.0]

| Starting from ``Low``, reaching ``High`` takes eight steps on average.
| Starting from ``Moderate``, it takes six steps on average.
| Starting from ``High``, the hitting time is zero.

| Let :math:`m_L` and :math:`m_M` denote the expected times to reach ``High`` from ``Low`` and ``Moderate``.
| Each transition uses one step, followed by the expected remaining time:

.. math::

   \begin{aligned}
   m_L &= 1 + 0.5m_L + 0.5m_M \\
   m_M &= 1 + 0.25m_L + 0.5m_M
   \end{aligned}

Solving these equations gives :math:`m_L = 8` and :math:`m_M = 6`.

First-Passage Probabilities
---------------------------

| A mean hitting time tells us how long reaching a target takes on average.
| The **first-passage probabilities** instead describe the chance of reaching it for the first time at each step.
| From ``Low``, traffic must pass through ``Moderate`` before reaching ``High``.

The first-passage probabilities for the first four steps are:

.. code-block:: python

   print(mc.first_passage_probabilities(4, 'Low', ['High']))

The result is:

.. code-block:: text

   [0.0000, 0.1250, 0.1250, 0.1094]

| Reaching ``High`` on the first step is impossible.
| The probability of first reaching it on the second step is :math:`0.5 \times 0.25 = 0.125`.

Commute Times
-------------

| The **commute time** between two states is the expected time to travel from one to the other and then return.
| Unlike a hitting time, it describes a complete round trip.

PyDTMC computes the commute times for every pair of states:

.. code-block:: python

   print(mc.commute_times())

The resulting matrix is:

.. code-block:: text

   [[ 0.0, 8.0, 16.0]
    [ 8.0, 0.0,  8.0]
    [16.0, 8.0,  0.0]]

| Each entry represents the expected round-trip time between two states, following their order in the transition matrix.
| For example, the value ``16.0`` represents the commute time between ``Low`` and ``High``.
| This includes eight expected steps to reach ``High`` and another eight to return to ``Low``.
| The matrix is symmetric because the round-trip time is the same in either direction.

The relationship is expressed by:

.. math::

   \kappa_{ij} = m_{ij} + m_{ji}

Here, :math:`m_{ij}` is the mean first-passage time from state :math:`i` to state :math:`j`.