Markov Chain
============

| A discrete-time Markov chain is a stochastic process that describes transitions among a finite set of states.
| Its transition matrix specifies the probability of moving from each state (row) to another state (column) in a single step.
| Since these probabilities describe all possible transitions from a given state, each row of the matrix must sum to 1.

Instance Creation
-----------------

A simple two-state Markov chain with explicitly named states can be created as follows:

.. code-block:: python

   from pydtmc import MarkovChain

   p = [[0.7, 0.3], [0.2, 0.8]]
   mc = MarkovChain(p, ['A', 'B'])

Properties Inspection
---------------------

The properties of a Markov chain provide information about its structure and long-term behavior:

.. code-block:: python

   print(mc.is_irreducible)
   print(mc.is_aperiodic)
   print(mc.is_ergodic)
   print(mc.pi[0])

| It is **irreducible** because each state can be reached from the other.
| It is also **aperiodic** because both states have positive self-transition probabilities.
| Consequently, the chain is **ergodic**.

| Its unique **stationary distribution** is ``[0.4, 0.6]``.
| In the long run, the chain spends approximately 40% of its time in state ``A`` and 60% in state ``B``.

| The ``pi`` property returns a list of stationary distributions, one for each recurrent class.
| Since this chain is irreducible, it has only one stationary distribution that can be accessed using ``mc.pi[0]``.

Simulation
----------

A sequence of state transitions can be generated using the ``simulate`` method:

.. code-block:: python

   print(mc.simulate(10, seed=42))

| Specifying a fixed random seed makes the simulation reproducible.
| Short simulations may produce state frequencies that differ from the stationary distribution.
| As the simulation length increases, these frequencies converge to the stationary probabilities.

.. seealso::

   For a complete list of properties and methods, see the :doc:`MarkovChain API Reference <../api_reference/markov_chain>`.