Assessments
===========

| An observed sequence may resemble a Markov chain without following its assumptions exactly.
| Statistical assessments help us examine those assumptions and compare observations with a proposed model.
| This guide introduces the assessment functions available in PyDTMC using a simple weather process.

Model Construction
------------------

| Suppose the weather has two states, ``Dry`` and ``Wet``.
| The transition matrix describes how the weather changes from one day to the next.

.. math::

   P = \begin{pmatrix}
      0.8 & 0.2 \\
      0.3 & 0.7
   \end{pmatrix}

| A dry day is followed by another dry day with probability ``0.8``.
| A wet day is followed by another wet day with probability ``0.7``.

Using PyDTMC, we can create the model and generate a sequence of observations:

.. code-block:: python

   from pydtmc import MarkovChain

   states = ['Dry', 'Wet']
   mc = MarkovChain([[0.8, 0.2], [0.3, 0.7]], states)
   observations = mc.simulate(1000, initial_state='Dry', seed=42)

| The simulation contains the initial state and 1000 subsequent transitions.
| Using a known model lets us examine the assessments under controlled conditions.
| Statistical tests may still reject a true hypothesis because of random variation.
| In a real application, the sequence could instead come from recorded weather observations.

Understanding Statistical Assessments
-------------------------------------

| Most assessments begin with a **null hypothesis**, describing a property that the observations are assumed to satisfy.
| A **test statistic** summarizes how strongly the observations differ from what the null hypothesis predicts.
| The **p-value** measures how unusual a test statistic at least as extreme as the observed one would be if the null hypothesis were true.
| We compare the p-value with a chosen **significance level**, usually ``0.05``.

| A p-value below ``0.05`` provides evidence against the null hypothesis.
| A larger p-value means that the assessment has not found sufficient evidence to reject it.
| It does not prove that the hypothesis is true.

| Most PyDTMC assessments return a rejection decision, a p-value, and additional test information.
| A rejection decision of ``True`` means that the null hypothesis was rejected.
| Some tests can return ``None`` when a decision cannot be determined from the observations.

Independence & Markov Property
------------------------------

| Independence means that knowing today's weather does not help predict tomorrow's weather.
| The ``assess_independence`` function tests this assumption using observed transitions.

.. code-block:: python

   from pydtmc import assess_independence

   rejected, p_value, _ = assess_independence(states, observations)

   print(rejected)
   print(p_value)

| Rejecting independence suggests that consecutive weather observations are related.
| However, dependence alone does not establish the Markov property.

| The first-order Markov property means that tomorrow's weather depends on today's weather, not on earlier days once today's weather is known.
| The ``assess_markov_property`` function checks whether the observed sequence is compatible with this assumption.

.. code-block:: python

   from pydtmc import assess_markov_property

   rejected, p_value, _ = assess_markov_property(states, observations)

   print(rejected)
   print(p_value)

| Here, a rejection suggests that the immediately preceding state may not contain enough information to describe the next transition.
| A non-rejection means that the observations do not provide sufficient evidence against a first-order model.

Assessing Markov Order
----------------------

| A first-order model uses one previous state to predict the next state.
| A higher-order model can use several previous states instead.
| The ``assess_order`` function compares candidate orders and selects one using an information criterion.
| By default, it uses the **Bayesian Information Criterion (BIC)**, which balances how well a model explains the observations against its complexity.

.. code-block:: python

   from pydtmc import assess_order

   order, scores = assess_order(states, observations, maximum_order=3)

   print(order)
   print(scores)

| The returned ``order`` is the preferred order among ``0``, ``1``, ``2``, and ``3``.
| Order ``0`` represents an independent process, while order ``1`` represents an ordinary first-order Markov chain.
| The ``scores`` dictionary gives the criterion value for each candidate; smaller values are preferred.

Time Homogeneity & Change Points
--------------------------------

| A Markov chain is time-homogeneous when its transition probabilities do not change over time.
| For example, the probability of moving from ``Dry`` to ``Wet`` should follow the same rule throughout the observation period.

| The ``assess_time_homogeneity`` function divides the observed transitions into consecutive blocks and compares their behavior.

.. code-block:: python

   from pydtmc import assess_time_homogeneity

   rejected, p_value, _ = assess_time_homogeneity(states, observations, blocks=2)

   print(rejected)
   print(p_value)

| A rejection suggests that the transition probabilities differ between the two parts of the sequence.
| A non-rejection does not guarantee that they were perfectly constant.

| Instead of comparing predefined time blocks, we may want to locate an abrupt change in transition probabilities.
| The ``assess_change_point`` function searches for a possible transition between two regimes.

.. code-block:: python

   from pydtmc import assess_change_point

   change_point, p_value, _ = assess_change_point(states, observations, simulations=200, seed=42)

   print(change_point)
   print(p_value)

| If a statistically significant change is detected, ``change_point`` identifies the first transition of the new regime, using zero-based indexing.
| Otherwise, the returned change point is ``None``.
| The p-value is estimated from simulated sequences, so the number of simulations affects its precision.

Comparing Observation Sequences
-------------------------------

| We may also collect two separate weather records and ask whether they follow the same transition rules.
| The ``assess_homogeneity`` function compares the transition probabilities across multiple sequences.

.. code-block:: python

   from pydtmc import assess_homogeneity

   first = mc.simulate(1000, initial_state='Dry', seed=42)
   second = mc.simulate(1000, initial_state='Wet', seed=43)

   rejected, p_value, _ = assess_homogeneity(states, [first, second])

   print(rejected)
   print(p_value)

| A rejection suggests that the sequences do not share the same transition probabilities.
| The test concerns their transition rules, not whether they start in the same state.

Theoretical Model Compatibility
-------------------------------

| If a transition matrix is already available, we can ask whether observed transitions agree with it.
| The ``assess_theoretical_compatibility`` function compares the sequence with the probabilities in the proposed Markov chain.

.. code-block:: python

   from pydtmc import assess_theoretical_compatibility

   rejected, p_value, _ = assess_theoretical_compatibility(mc, observations)

   print(rejected)
   print(p_value)

| A rejection provides evidence that the observed transitions differ from the proposed model.
| A non-rejection means that the observed differences are not sufficient to reject it at the chosen significance level.

Assessing Stationary Initialization
-----------------------------------

| Time homogeneity means that transition probabilities remain constant over time.
| Stationarity means that the distribution of the states remains constant over time.
| A time-homogeneous chain need not start in its stationary distribution.
| The ``assess_stationarity`` function tests whether independent sequences were initialized according to that distribution.

We can generate independent sequences with initial states drawn from the stationary distribution:

.. code-block:: python

   import numpy as np
   from pydtmc import assess_stationarity

   rng = np.random.default_rng(42)
   starts = rng.choice(states, size=100, p=mc.pi[0])

   sequences = [
       mc.simulate(10, initial_state=start, seed=i)
       for i, start in enumerate(starts)
   ]

   rejected, p_value, _ = assess_stationarity(mc, sequences)

   print(rejected)
   print(p_value)

| This assessment compares the initial states of the sequences with the model's stationary distribution.
| It does not test whether one observed sequence has constant transition probabilities.
| As with the other assessments, a non-rejection does not prove that the starting distribution is stationary.