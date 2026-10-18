from pathlib import Path
from re import findall

from pydtmc import MarkovChain


class TextGenerator:  # pylint: disable=missing-class-docstring

    _separator = '<sep>'

    def __init__(self, corpus: str):

        sentences = [findall(r'[a-z]+', line.lower()) for line in corpus.splitlines()]
        sentences = [sentence for sentence in sentences if sentence]

        if len(sentences) < 2:
            raise ValueError('The corpus must contain at least two non-empty lines.')

        tokens = [self._separator]

        for sentence in sentences:
            tokens.extend(sentence)
            tokens.append(self._separator)

        tokens.append(sentences[0][0])

        pairs = list(zip(tokens, tokens[1:]))
        unique_pairs = list(dict.fromkeys(pairs))

        if len(unique_pairs) < 2:
            raise ValueError('The corpus must contain at least two distinct word pairs.')

        labels = [f's{i}' for i in range(len(unique_pairs))]
        pair_to_label = dict(zip(unique_pairs, labels))

        self._label_to_pair = dict(zip(labels, unique_pairs))
        self._initial_state = labels[0]
        self._chain = MarkovChain.fit_sequence('mle', labels, [pair_to_label[pair] for pair in pairs])

    @property
    def initial_state(self) -> str:
        return self._initial_state

    @property
    def states(self) -> int:
        return len(self._label_to_pair)

    def generate(self, words: int = 80, seed: int = 42) -> str:

        if words < 4:
            raise ValueError('The number of words must be at least four.')

        states = self._chain.simulate(max(100, words * 8), initial_state=self._initial_state, seed=seed)

        previous_pair = self._label_to_pair[states[0]]
        sentence = [previous_pair[1]]

        output, emitted = [], 1

        for label in states[1:]:

            pair = self._label_to_pair[label]

            if previous_pair[1] != pair[0]:
                raise RuntimeError('The fitted chain generated incompatible word pairs.')

            previous_pair = pair
            token = pair[1]

            if token == self._separator:

                if sentence:
                    output.append(' '.join(sentence).capitalize() + '.')
                    sentence = []

            else:

                sentence.append(token)
                emitted += 1

                if emitted == words:
                    output.append(' '.join(sentence).capitalize() + '.')
                    return ' '.join(output)

        raise RuntimeError('Insufficient generated transitions to reach the requested word count.')

    @classmethod
    def from_file(cls, path: str | Path) -> 'TextGenerator':

        path = path if isinstance(path, str) else Path(path)
        instance = cls(path.read_text(encoding='utf-8'))

        return instance
