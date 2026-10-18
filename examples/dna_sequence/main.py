from argparse import ArgumentParser
from pathlib import Path

from model import SequenceModel  # pylint: disable=import-error


def main() -> None:

    parser = ArgumentParser(description='Model and generate synthetic DNA sequences with PyDTMC.')
    parser.add_argument('--sequence', type=Path, default=Path(__file__).with_name('sequence.txt'))
    parser.add_argument('--length', type=int, default=120, help='The number of nucleotides to generate.')
    parser.add_argument('--seed', type=int, default=42, help='The random seed.')
    args = parser.parse_args()

    model = SequenceModel.from_file(args.sequence)

    observed = model.nucleotide_frequencies(model.sequence)
    observed_pairs = model.dinucleotide_frequencies(model.sequence)

    generated = model.generate(args.length, args.seed)
    simulated = model.nucleotide_frequencies(generated)
    simulated_pairs = model.dinucleotide_frequencies(generated)

    stationary_distribution = model.stationary_distribution

    print(f'Observed nucleotides: {len(model.sequence)}')
    print('')
    print(f'Generated nucleotides: {len(generated)}')
    print(f'Generated sequence: {generated}')
    print('')
    print('Nucleotide frequencies (observed / generated / stationary):')

    for nucleotide in model.nucleotides:
        print(f'  {nucleotide}: {observed[nucleotide]:.2%} / {simulated[nucleotide]:.2%} / {stationary_distribution[nucleotide]:.2%}')

    print('Dinucleotide frequencies (observed / generated):')

    for pair in observed_pairs:
        print(f'  {pair}: {observed_pairs[pair]:.2%} / {simulated_pairs[pair]:.2%}')


if __name__ == '__main__':
    main()
