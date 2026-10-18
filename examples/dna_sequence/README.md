# DNA Sequence

This standalone example uses PyDTMC to fit and generate synthetic DNA-like nucleotide sequences with a first-order Markov chain.\
It compares observed and generated nucleotide and dinucleotide frequencies, along with the fitted chain's stationary distribution.\
The model assumes that each nucleotide depends only on its immediate predecessor.

With PyDTMC installed, open a terminal in the example directory:

```bash
# Run the example with the included synthetic sequences.
python main.py

# Generate 500 nucleotides using a fixed random seed, using a custom nucleotide sequence.
python main.py --sequence path/to/sequence.txt --length 500 --seed 7
```

The synthetic `sequence.txt` contains 1,200 nucleotides among those available: `A`, `C`, `G`, and `T`.\
Lines are wrapped for readability; line breaks and other whitespace are ignored.\
Lowercase characters are accepted and handled, but ambiguous symbols such as `N` are not.

The `SequenceModel` class:
1. Reads and validates the nucleotide sequence.
2. Fits a first-order PyDTMC `MarkovChain` from consecutive nucleotides.
3. Generates a reproducible sequence using `MarkovChain.simulate()`.
4. Computes observed and generated nucleotide and dinucleotide frequencies.
5. Calculates the fitted chain's stationary nucleotide frequencies.

The following metrics are calculated:
- **Generated sequence:** a new nucleotide sequence sampled from the fitted model.
- **Nucleotide frequencies:** fractions of `A`, `C`, `G`, and `T` in each sequence.
- **Dinucleotide frequencies:** fractions of each consecutive nucleotide pair.
- **Stationary frequencies:** long-run nucleotide probabilities under the fitted chain.

A first-order model preserves only one-step conditional structure.\
It does not model codons, biological constraints, or long-range sequence dependencies.\
The first generated nucleotide is fixed to the training sequence's initial nucleotide.