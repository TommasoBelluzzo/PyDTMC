# Regime switching

**Goal:** Fit and interpret a discrete-observation hidden Markov model for synthetic regime-dependent observations.

**Planned inputs:** A reproducible sequence of discrete symbols generated from a known model.

**Planned implementation:** `main.py` for fitting and evaluating a `pydtmc.HiddenMarkovModel`; optional `model.py` for input encoding and result interpretation.

**Modelling requirements:** Use discrete emission symbols supported by the existing HMM API. Do not imply that Gaussian-emission HMMs are already supported. Compare recovered quantities with the known synthetic process where appropriate, allowing for latent-label permutation.
