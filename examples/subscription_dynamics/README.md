# Subscription Dynamics

This standalone example uses PyDTMC to analyze subscription lifecycles through an absorbing Markov chain.\
It estimates customer retention, churn probabilities, expected subscription lifetimes, and discounted lifetime contributions.\
The model assumes that monthly transition probabilities depend only on the current subscription state and remain constant over time.\
It does not take into account reactivation after churn, incomplete customer histories, acquisition costs, or heterogeneous customer cohorts.

With PyDTMC installed, open a terminal in the example directory:

```bash
# Run the example with the included synthetic dataset.
python main.py

# Analyze a custom dataset over an eighteen-month horizon and with a 0.02 discount rate.
python main.py --customers path/to/customers.csv --months 18 --discount 0.02
```

The synthetic `customers.csv` contains monthly histories for 200 customers, with three columns:
- `customer_id`: unique customer identifier;
- `month`: consecutive observation index, starting at `0`;
- `state`: `active`, `at_risk`, `dormant`, or `churned`.

Each history starts in `active` and ends in `churned`. The file includes the terminal churn observation. All months are equal-length modelling steps, not calendar months.

The `RetentionModel` class:
1. Reads and validates complete customer histories.
2. Counts transitions within each customer history, never between customers.
3. Estimates transition probabilities and makes `churned` absorbing.
4. Constructs a PyDTMC `MarkovChain` to represent the subscription lifecycle.
5. Calculates finite-horizon churn, expected remaining lifetime, and discounted contributions.

The following metrics are calculated:
- **Observed mean lifetime:** average monthly transitions before churn in the sample.
- **Model mean time to churn:** expected transitions until churn, starting from `active`.
- **Churn probability:** probability of reaching `churned` within the selected horizon.
- **Retention probability:** probability of remaining unchurned at that horizon.
- **Discounted lifetime contribution:** expected discounted contribution until churn, including the starting month.
- **Retention by state:** horizon-specific survival probability from each non-terminal state.

For illustration, monthly contributions are 20 units in `active`, 12 in `at_risk`, and zero otherwise.\
The default discount rate is 1% per month.