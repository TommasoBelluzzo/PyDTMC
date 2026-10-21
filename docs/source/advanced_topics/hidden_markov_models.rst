Hidden Markov Models
====================

| A hidden Markov model (HMM) describes transitions among hidden states that produce observable symbols.
| Its transition matrix gives the probabilities of moving between hidden states.
| Its emission matrix gives the probabilities of observing each symbol from a hidden state.
| This guide explores how to infer an unobserved process from the observations it produces.

Model Construction
------------------

| Suppose there are two weather conditions, ``Dry`` and ``Wet``, that cannot be directly observed.
| Instead, we observe either ``No Umbrella`` or ``Umbrella``.
| A **hidden Markov model** combines transitions between hidden states with probabilities of observing each symbol.

The transition matrix describes how the weather changes:

.. math::

   P = \begin{pmatrix}
      0.8 & 0.2 \\
      0.3 & 0.7
   \end{pmatrix}

| Rows and columns follow the order ``Dry`` and ``Wet``.
| For example, wet weather is followed by wet weather with probability ``0.7``.

The **emission matrix** describes the observations produced by each weather condition:

.. math::

   E = \begin{pmatrix}
      0.9 & 0.1 \\
      0.2 & 0.8
   \end{pmatrix}

| Rows represent ``Dry`` and ``Wet``; columns represent ``No Umbrella`` and ``Umbrella``.
| When the weather is wet, an umbrella is observed with probability ``0.8``.
| Observing an umbrella does not guarantee wet weather, because it can also appear on a dry day.

Using PyDTMC, the model can be created as follows:

.. code-block:: python

   from pydtmc import HiddenMarkovModel

   p = [
       [0.8, 0.2],
       [0.3, 0.7]
   ]

   e = [
       [0.9, 0.1],
       [0.2, 0.8]
   ]

   hmm = HiddenMarkovModel(
       p, e,
       ['Dry', 'Wet'],
       ['No Umbrella', 'Umbrella']
   )

Simulating Observations
-----------------------

| A simulation generates both the hidden weather conditions and their observable symbols.
| In a real application, we would typically see only the observations.

.. code-block:: python

   states, symbols = hmm.simulate(10, initial_state='Dry', seed=42)

   print(states)
   print(symbols)

| The simulation includes the initial state and its observation, followed by ten transitions.
| The two sequences correspond step by step.
| The hidden states describe what happened, while the symbols describe what was observed.
| Different observations can be produced by the same hidden state.

Observation Probabilities
-------------------------

Suppose we observe the following sequence:

.. code-block:: python

   observations = ['Umbrella', 'Umbrella', 'No Umbrella']

| We want to know how likely this sequence is under the model.
| The ``decode`` method accounts for all possible hidden-state sequences and returns the logarithm of the observation probability.
| Applying ``np.exp`` converts the logarithm back to an ordinary probability.

.. code-block:: python

   import numpy as np

   log_probability, posterior, _, _, _, _ = hmm.decode(
       observations,
       initial_status=[0.5, 0.5]
   )

   print(round(np.exp(log_probability), 4))

The observation probability is:

.. code-block:: text

   0.1073

| This is the probability of observing all three symbols in the specified order.
| It is not the probability that any particular hidden-state sequence occurred.

Inferring Hidden States
-----------------------

| After observing the umbrella sequence, we can estimate the probability of each hidden weather condition.
| These are **posterior probabilities**: they use all three observations, rather than only the symbol at one step.

.. code-block:: python

   print(np.round(posterior, 4))

The posterior probabilities are:

.. code-block:: text

   [[0.0589, 0.1133, 0.6913]
    [0.9411, 0.8867, 0.3087]]

| Rows correspond to ``Dry`` and ``Wet``, while columns follow the observations in chronological order.
| The probability of wet weather at the first observation is approximately 94.11%.
| At the final observation, dry weather is more likely, with probability approximately 69.13%.

Predicting State Sequences
--------------------------

| Posterior probabilities describe the likely state at each observation separately.
| We can also ask for a single hidden-state sequence that best explains all the observations.
| The **Viterbi algorithm** finds the most probable complete sequence.

.. code-block:: python

   _, path = hmm.predict(
       'viterbi',
       observations,
       initial_status=[0.5, 0.5]
   )

   print(path)

The most probable hidden-state sequence is:

.. code-block:: text

   ['Wet', 'Wet', 'Dry']

| The first two umbrella observations favor wet weather, while the final observation favors dry weather.
| Unlike choosing the most likely state independently at each step, Viterbi considers the transitions between states when selecting the complete path.