Hidden Markov Model
===================

| A hidden Markov model (HMM) describes transitions among hidden states that produce observable symbols.
| Its transition matrix gives the probabilities of moving between hidden states.
| Its emission matrix gives the probabilities of observing each symbol from a hidden state.
| Both matrices must have rows that sum to 1.

Instance Creation
-----------------

A two-state hidden Markov model with four observable symbols can be created as follows:

.. code-block:: python

   from pydtmc import HiddenMarkovModel

   p = [[0.4, 0.6], [0.8, 0.2]]
   e = [[0.5, 0.0, 0.0, 0.5], [0.2, 0.2, 0.2, 0.4]]
   hmm = HiddenMarkovModel(p, e, ['A', 'B'], ['H1', 'H2', 'H3', 'H4'])

Decoding
--------

The model can evaluate emission probabilities and infer hidden-state probabilities from observed symbols:

.. code-block:: python

   print(hmm.emission_probability('H1', 'A'))

   log_probability, posterior, _, _, _, _ = hmm.decode(['H1', 'H2', 'H3'])
   print(posterior)

| State ``A`` emits symbol ``H1`` with probability ``0.5``.
| Decoding estimates the probability of each hidden state at each observation.
| These states are inferred rather than directly observed.

Simulation
----------

A sequence of hidden states and observable symbols can be generated using the ``simulate`` method:

.. code-block:: python

   print(hmm.simulate(12, seed=42))

| Specifying a fixed random seed makes the simulation reproducible.
| The generated observations depend on the hidden states and their emission probabilities.

.. seealso::

   For a complete list of properties and methods, see the :doc:`HiddenMarkovModel API Reference <../api_reference/hidden_markov_model>`.