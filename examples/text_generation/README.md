# Text Generation

This standalone example uses PyDTMC's `MarkovChain.fit_sequence()` and `MarkovChain.simulate()` to generate "lorem ipsum" style placeholder text.\
The included corpus is an original, synthetic collection of short, Latin-like sentences.

With PyDTMC installed, open a terminal in the example directory:

```bash
# Run the example with default settings.
python examples/text_generation/main.py

# Generate 120 words using a fixed random seed.
python examples/text_generation/main.py --words 120 --seed 7
```

Each non-empty line in `corpus.txt` represents one sentence. The code:
1. lowercases and tokenizes its words;
2. inserts sentence-boundary marker*
3. transforms every consecutive pair `(word[t-1], word[t])` into a state of an ordinary first-order `MarkovChain`.

Thus, the generated next word is conditioned on the preceding **two** words.\
Increasing the order improves local context but may reproduce longer training fragments.

The sequence loops back to its starting pair, ensuring every state has an outgoing transition.\
Sentence-boundary markers prevent artificial transitions between sentences and support output formatting.