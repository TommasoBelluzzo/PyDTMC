from collections import Counter
from pathlib import Path

from pydtmc import MarkovChain


class SequenceModel:  # pylint: disable=missing-class-docstring

    _nucleotides = ['A', 'C', 'G', 'T']

    def __init__(self, sequence: str):

        if len(sequence) < 4:
            raise ValueError('The sequence must contain at least four nucleotides.')

        if any(symbol not in self._nucleotides for symbol in sequence):
            raise ValueError('The sequence must contain only: A, C, G, and T.')

        chain = MarkovChain.fit_sequence('mle', self._nucleotides, list(sequence))

        if not chain.is_irreducible:
            raise ValueError('The observed sequence must form an irreducible chain.')

        probabilities = chain.pi[0]
        stationary_distribution = dict(zip(self._nucleotides, map(float, probabilities)))

        self._chain = chain
        self._sequence = sequence
        self._stationary_distribution = stationary_distribution

    @property
    def nucleotides(self) -> list[str]:
        return self._nucleotides.copy()

    @property
    def sequence(self) -> str:
        return self._sequence

    @property
    def stationary_distribution(self) -> dict[str, float]:
        return self._stationary_distribution.copy()

    def generate(self, length: int = 120, seed: int = 42) -> str:

        if length < 4:
            raise ValueError('The generated sequence must have at least four nucleotides.')

        states = self._chain.simulate(length - 1, initial_state=self.sequence[0], seed=seed)
        sequence = ''.join(states)

        return sequence

    @classmethod
    def from_file(cls, path: str | Path) -> 'SequenceModel':

        path = path if isinstance(path, str) else Path(path)

        text = path.read_text(encoding='utf-8')
        sequence = ''.join(text.split()).upper()

        instance = cls(sequence)

        return instance

    @staticmethod
    def dinucleotide_frequencies(sequence: str) -> dict[str, float]:

        pairs = [sequence[index:index + 2] for index in range(len(sequence) - 1)]
        counts = Counter(pairs)

        value = {
            first + second: counts[first + second] / len(pairs)
            for first in SequenceModel._nucleotides
            for second in SequenceModel._nucleotides
        }

        return value

    @staticmethod
    def nucleotide_frequencies(sequence: str) -> dict[str, float]:

        counts = Counter(sequence)
        value = {symbol: counts[symbol] / len(sequence) for symbol in SequenceModel._nucleotides}

        return value
