# Monopoly Board Analysis

This standalone example uses PyDTMC to analyze movement around a simplified Monopoly board with a finite Markov chain.\
It estimates long-run square probabilities, expected visits, and simulated movements over the board.

This is a movement-only approximation, not a full Monopoly rules engine.\
Each step represents one roll of two fair six-sided dice, rather than a complete player turn under official rules.\
Doubles do not grant extra rolls or trigger three-doubles jail, and arriving at Jail does not cause detention.\
There are no property purchases, rents, bankruptcies, or strategies.\
Chance and Community Chest each draw uniformly from a 16-card abstraction on every visit; cards are not tracked or depleted.\
Only movement effects are modelled, and card movements do not trigger further card draws. These conventions affect the probabilities.

With PyDTMC installed, open a terminal in the example directory:

```bash
# Run the example with the included board configuration.
python main.py

# Compare expected visits over 250 rolls using a fixed simulation seed.
python main.py --rolls 250 --seed 7

# Load a board file with alternative square names.
python main.py --board path/to/board.json
```

The included `board.json` lists the 40 squares of the classic US board in clockwise order, starting at `GO`.\
Square names are used for presentation; movement rules and special-square indices are defined in `model.py`.

The `MonopolyModel` class:
1. Loads and validates the board configuration.
2. Constructs a transition matrix from the probabilities of two dice.
3. Applies Go to Jail and randomized Chance and Community Chest movements.
4. Constructs a PyDTMC `MarkovChain` and computes its stationary distribution.
5. Estimates finite-horizon square visits and simulates a sequence of positions.

The following metrics are calculated:
- **Stationary probability:** the long-run proportion of rolls ending on each square.
- **Expected visits:** expected number of times a roll ends on each square, starting at `GO` and excluding the initial position.
- **Simulated positions:** a reproducible sequence of model positions.