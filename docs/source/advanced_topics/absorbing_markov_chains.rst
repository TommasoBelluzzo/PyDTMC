Absorbing Markov Chains
=======================

| This guide explores how a Markov chain can describe a process that eventually reaches a final outcome.
| The example follows an application through submission, review, approval, or rejection.

Model Construction
------------------

| An application can remain ``Submitted``, move to ``Review``, or be ``Approved`` immediately.
| During review, it may remain under consideration, be approved, or be rejected.
| Once approved or rejected, the application does not change status.

The transition matrix is:

.. math::

   P = \begin{pmatrix}
       0.50 & 0.25 & 0.25 & 0.00 \\
       0.00 & 0.50 & 0.25 & 0.25 \\
       0.00 & 0.00 & 1.00 & 0.00 \\
       0.00 & 0.00 & 0.00 & 1.00
   \end{pmatrix}

| Rows and columns follow the order ``Submitted``, ``Review``, ``Approved``, and ``Rejected``.
| For example, an application under review has a 50% chance of remaining under review for another step.
| It also has a 25% chance of approval and a 25% chance of rejection.

Using PyDTMC, the chain can be created as follows:

.. code-block:: python

   from pydtmc import MarkovChain

   p = [
       [0.50, 0.25, 0.25, 0.00],
       [0.00, 0.50, 0.25, 0.25],
       [0.00, 0.00, 1.00, 0.00],
       [0.00, 0.00, 0.00, 1.00]
   ]

   mc = MarkovChain(p, ['Submitted', 'Review', 'Approved', 'Rejected'])

Absorbing and Transient States
------------------------------

| An **absorbing state** is a state that cannot be left once entered.
| Here, ``Approved`` and ``Rejected`` are absorbing states: both have a probability of ``1.0`` of remaining unchanged.
| A **transient state** is one that the chain may leave without ever returning to it.
| ``Submitted`` and ``Review`` are transient because the application can eventually reach a final decision.

| A chain is **absorbing** when it has at least one absorbing state and every state can reach an absorbing state.
| In this model, every application eventually reaches approval or rejection with probability 1.

These properties can be inspected as follows:

.. code-block:: python

   print(mc.absorbing_states)
   print(mc.transient_states)
   print(mc.is_absorbing)

Absorption Probabilities
------------------------

| Reaching a final decision is certain, but the outcome is not.
| The **absorption probabilities** describe the chances of eventually reaching each absorbing state.
| An application under review has equal chances of eventual approval and rejection.
| A submitted application has a higher chance of approval because it may be approved without entering review.

Using PyDTMC, these probabilities can be computed as follows:

.. code-block:: python

   print(mc.absorption_probabilities())

The resulting matrix is:

.. math::

   B = \begin{pmatrix}
       0.75 & 0.50 \\
       0.25 & 0.50
   \end{pmatrix}

| Each row represents an absorbing state, ordered as ``Approved`` and ``Rejected``.
| Each column represents a transient starting state, ordered as ``Submitted`` and ``Review``.
| Thus, a submitted application is eventually approved with probability ``0.75`` and rejected with probability ``0.25``.
| An application already under review has a ``0.50`` probability of either outcome.

Expected Absorption Times
-------------------------

| The **absorption time** is the number of transitions needed to reach a final decision.
| It can vary from one application to another, therefore we calculate its expected value.

.. code-block:: python

   print(mc.absorption_times('mean'))

The expected absorption times are:

.. math::

   \mathbf{t} = \begin{pmatrix} 3 \\ 2 \end{pmatrix}

| A submitted application takes an average of three transitions to reach a final decision.
| An application already under review takes an average of two transitions.

Occupation Matrix
-----------------

| Before a final decision, an application may spend several steps in a transient state.
| The **occupation matrix** gives the expected number of visits to each transient state before absorption.
| It includes the starting state as a visit.

.. code-block:: python

   print(mc.occupation_matrix)

For this model, the occupation matrix is:

.. math::

   N = \begin{pmatrix}
      2 & 1 \\
      0 & 2
   \end{pmatrix}

| Rows represent the starting states and columns represent the states visited, both ordered as ``Submitted`` and ``Review``.
| Starting from ``Submitted``, an application spends an expected two steps there and one step in ``Review``.
| Starting from ``Review``, it spends an expected two steps under review and never returns to ``Submitted``.
| The row sums, ``3`` and ``2``, agree with the expected absorption times calculated above.

Simulation
----------

A simulation generates one possible sequence of application statuses:

.. code-block:: python

   print(mc.simulate(10, initial_state='Submitted', seed=42))

| The simulated application may remain in a transient state for several steps before reaching a final decision.
| Once it reaches ``Approved`` or ``Rejected``, its status cannot change.
| Unlike the expected absorption times, a simulation shows one possible outcome rather than an average over many applications.