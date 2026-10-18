# Server Reliability

This standalone example uses PyDTMC to analyze the availability of a server with a discrete-time Markov chain.\
It estimates long-run operational health, failure-state occupancy, and expected available hours from a synthetic service-health log.\
The model assumes that hourly service states depend only on the preceding state and that transition probabilities do not change over time.

With PyDTMC installed, open a terminal in the example directory:

```bash
# Run the example with the included synthetic dataset.
python main.py

# Analyze a custom hourly log over a 48-hour horizon.
python main.py --incidents path/to/incidents.csv --hours 48
```

The synthetic `incidents.csv` contains consecutive hourly observations during January 2025, with two columns:
- `timestamp`: an hourly timestamp in the strict format `YYYY-MM-DD HH:MM`;
- `state`: one of `operational`, `degraded`, `failed`, or `recovering`.

The `operational` and `degraded` states count as available; the `failed` and `recovering` states count as unavailable.\
Each observation describes one hour-long service interval. The dataset is fictional, not a record of real incidents.

The `ReliabilityModel` class:
1. Reads and validates the ordered, consecutive hourly observations.
2. Estimates a PyDTMC `MarkovChain` from observed state transitions.
3. Requires an irreducible fitted chain for a unique stationary distribution.
4. Calculates stationary availability and failure-state occupancy.
5. Uses `recurrence_times('mean')` and `expected_rewards()` for recurrence and a finite-horizon availability measure.

The following metrics are calculated:
- **Long-run availability:** the stationary probability of being `operational` or `degraded`.
- **Long-run failed fraction:** the stationary probability of being `failed`.
Mean failure-state recurrence: expected hours between visits to failed; vonsecutive failed hours count as separate visits, not distinct outages.
Expected available hours: expected available observations over the selected horizon, starting from operational; the initial observation is included.
- **Stationary distribution:** the long-run probability of each service state.

The reported values depend on the fitted first-order, time-homogeneous model.\
They do not account for scheduled maintenance, changing traffic, or nonstationary failure rates.\
Availability is a modelled state fraction, not a service-level agreement measurement.