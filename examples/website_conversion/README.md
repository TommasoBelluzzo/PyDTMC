# Website Conversion

This standalone example uses PyDTMC to analyze website visitor journeys through an absorbing Markov chain.\
It estimates conversion and abandonment probabilities, along with the expected number of transitions before a visitor reaches a terminal outcome.\
The analysis also assumes that transition probabilities depend only on the current page, not on the visitor's earlier navigation history.

With PyDTMC installed, open a terminal in the example directory:

```bash
# Run the example with the included synthetic dataset.
python main.py

# Run the example with a custom dataset.
python main.py --sessions path/to/sessions.csv
```

The synthetic `sessions.csv` contains complete visitor journeys across seven states:
- `home`: the starting page of each session;
- `product`: a product detail page;
- `search`: a product search page;
- `cart`: the shopping cart;
- `checkout`: the checkout page;
- `purchase` and `exit` as terminal outcomes.

The `ConversionModel` class:
1. Reads and validates the visitor sessions.
2. Counts transitions within each session, without connecting separate visitors.
3. Estimates transition probabilities from the observed frequencies.
4. Makes `purchase` and `exit` absorbing states.
5. Constructs a PyDTMC `MarkovChain` and computes conversion statistics.

The following metrics are calculated:
- **Observed conversion rate:** the fraction of sessions ending in a purchase.
- **Model conversion probability:** the probability of eventually reaching `purchase` from `home`.
- **Model abandonment probability:** the probability of eventually reaching `exit` from `home`.
- **Expected steps to outcome:** the expected number of transitions before reaching either terminal state.
- **Conversion probability by page:** the eventual purchase probability when starting from each non-terminal page.

The observed conversion rate and model conversion probability need not agree, since they describe different quantities.