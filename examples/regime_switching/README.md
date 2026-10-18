# Regime Switching

This standalone example uses PyDTMC to identify hidden market regimes from synthetic daily returns using a Hidden Markov Model (HMM).\
It discretizes returns into observable symbols, fits the model with Baum–Welch estimation, and infers regimes using Viterbi decoding.\
The model assumes that the next hidden regime depends only on the current regime and that observations depend only on the current hidden regime.

With PyDTMC installed, open a terminal in the example directory:

```bash
# Run the example with the included synthetic dataset.
python main.py

# Run the example with a custom dataset.
python main.py --observations path/to/observations.csv
```

The synthetic `observations.csv` contains daily returns for all weekdays in 2022, with two columns:
- `date`: a business weekday, strictly formatted as `YYYY-MM-DD`;
- `return`: a finite decimal daily return (for example, `0.012500` represents 1.25%).

Weekdays are used as a simple business-day calendar; dates must be strictly increasing and unique.

The `RegimeModel` class:
1. Reads and validates the observations and their dates.
2. Maps returns below -0.4% to `negative`, above +0.4% to `positive`, and all other returns to `neutral`.
3. Fits a three-state PyDTMC `HiddenMarkovModel` with Baum–Welch estimation and fixed initial parameters.
4. Uses Viterbi decoding to infer the most probable hidden-regime sequence.
5. Reports the estimated transition and emission matrices and the inferred regime changes.

The following metrics are calculated:
- **Transition matrix:** estimated probabilities of moving between hidden regimes.
- **Emission matrix:** estimated probabilities of observing each return category under each regime.
- **Inferred regime changes:** the number of changes between consecutive Viterbi states.
- **Recent inferred regimes:** the last ten dated observations and their inferred states.

The hidden states are labelled `regime_1`, `regime_2`, and `regime_3`; their interpretation is not known in advance and may change across fits. The data is synthetic, not historical market data. Discretization discards return magnitudes, and inferred regimes are model estimates rather than directly observed market conditions.
