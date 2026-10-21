Mixing & Convergence
====================

| This guide explains how quickly a Markov chain loses the influence of its starting state.
| A simple service-demand model illustrates convergence, mixing times, and related diagnostics.
| The model has three states: ``Low``, ``Normal``, and ``High``.

Model Construction
------------------

Suppose demand remains at its current level with probability ``0.6``.
It moves to either of the other levels with probability ``0.2``:

.. math::

   P = \begin{pmatrix}
      0.6 & 0.2 & 0.2 \\
      0.2 & 0.6 & 0.2 \\
      0.2 & 0.2 & 0.6
   \end{pmatrix}

Using PyDTMC, the chain can be created as follows:

.. code-block:: python

   from pydtmc import MarkovChain

   p = [
       [0.6, 0.2, 0.2],
       [0.2, 0.6, 0.2],
       [0.2, 0.2, 0.6]
   ]

   mc = MarkovChain(p, ['Low', 'Normal', 'High'])

| Every state can reach every other state in one step, so the chain is **irreducible**.
| Each state can also remain unchanged, making the chain **aperiodic** and therefore **ergodic**.
| Its unique stationary distribution assigns equal probability to all three demand levels.

.. code-block:: python

   print(mc.pi[0])

.. math::

   \pi = \left(\frac{1}{3},\frac{1}{3},\frac{1}{3}\right)

Convergence of State Probabilities
----------------------------------

| If demand starts at ``Low``, that state initially has probability ``1``.
| After each transition, the probabilities move closer to the stationary distribution.

.. code-block:: python

   import numpy as np

   for steps in [1, 2, 3]:
       print(np.round(mc.propagate(steps, 'Low'), 4))

The resulting distributions are:

.. code-block:: text

   [0.6000, 0.2000, 0.2000]
   [0.4400, 0.2800, 0.2800]
   [0.3760, 0.3120, 0.3120]

| The initial demand level becomes less important with each transition.
| In the long run, the probabilities approach one third for each state, regardless of where the chain starts.

Total Variation Distance
------------------------

| To measure convergence, we need a way to compare two probability distributions.
| The **total variation distance** measures their largest possible difference in probability over any set of states.
| For two distributions :math:`\mu` and :math:`\nu`, it is:

.. math::

   d_{\mathrm{TV}}(\mu,\nu) = \frac{1}{2}\sum_i |\mu_i-\nu_i|

For example, after three transitions from ``Low``, the distance to stationarity is:

.. math::

   d_{\mathrm{TV}} = \frac{1}{2}\left(\left|0.376-\frac{1}{3}\right| + 2\left|0.312-\frac{1}{3}\right|\right) \approx 0.0427

| A smaller distance means that the current state probabilities are closer to the stationary probabilities.
| A distance of zero means that the distributions are identical.

Mixing Time
-----------

| The **mixing time** is the number of transitions needed to become sufficiently close to stationarity.
| A chosen threshold specifies how small the total variation distance must be.
| PyDTMC measures the worst-case distance over all possible starting states.

For a threshold of ``0.05``:

.. code-block:: python

   print(mc.mixing_time(cutoff=0.05))

.. code-block:: text

   3

| After two transitions, the worst-case distance is approximately ``0.1067``.
| After three, it is approximately ``0.0427``, below the threshold.
| Thus, the mixing time is three transitions.

Contraction Coefficients
------------------------

| Contraction coefficients describe how much two different starting distributions can move closer after one transition.
| The **Dobrushin coefficient** is the greatest total variation distance between two rows of the transition matrix.
| The **Doeblin coefficient** measures the probability mass shared by every row.

.. math::

   \delta(P) = \frac{1}{2}\max_{i,j}\sum_k |P_{ik}-P_{jk}|

.. math::

   \alpha(P) = \sum_k\min_i P_{ik}

They can be inspected as follows:

.. code-block:: python

   print(mc.dobrushin_coefficient)
   print(mc.doeblin_coefficient)

.. code-block:: text

   0.4
   0.6000000000000001

| Here, the Dobrushin coefficient is ``0.4`` and the Doeblin coefficient is approximately ``0.6``.
| After one transition, the distance between any two state distributions is at most 40% of its original value.
| The general relationship is :math:`\delta(P) \leq 1-\alpha(P)`; equality holds for this example.

Spectral Gap & Relaxation
-------------------------

| The transition matrix has **eigenvalues**, which help describe how quickly state probabilities approach stationarity.
| In this model, the eigenvalues are ``1``, ``0.4``, and ``0.4``.
| The eigenvalue ``1`` represents the stationary distribution.
| The other eigenvalues determine how quickly deviations from stationarity decrease.

| In this case, these deviations are multiplied by ``0.4`` after each transition.
| This means that only 40% of the previous deviation remains after one step.

| The **absolute spectral gap** measures the separation between ``1`` and the largest modulus of the other eigenvalues:
| A larger spectral gap generally indicates faster convergence.

.. math::

   \gamma_{\mathrm{abs}} = 1-0.4 = 0.6

.. code-block:: python

   print(mc.absolute_spectral_gap)
   print(mc.relaxation_rate)

.. code-block:: text

   0.6
   1.6666666666666667

| The **relaxation rate** is the reciprocal of the absolute spectral gap, here approximately ``1.67``.
| It provides another measure of convergence speed: smaller values indicate faster relaxation toward stationarity.
| Unlike mixing time, it does not specify how many transitions are needed to reach a particular distance from the stationary distribution.

Conductance
-----------

| A **group of states** is any subset of the states in the Markov chain.
| The **conductance** measures how easily the chain moves from such a group to states outside it.
| For example, consider a group containing only ``Low``.
| The probability of moving from ``Low`` to either ``Normal`` or ``High`` in one step is ``0.4``.
| The conductance of this group is therefore ``0.4``.

.. code-block:: python

   print(mc.conductance(['Low']))

.. code-block:: text

   0.4

| The global conductance searches for the smallest conductance among eligible groups of states.
| For this symmetric three-state model, every single-state group has conductance ``0.4``.
| The result is therefore ``0.4``.

.. code-block:: python

   print(mc.global_conductance)

.. code-block:: text

   0.4