from argparse import ArgumentParser
from pathlib import Path

from generator import TextGenerator  # pylint: disable=import-error


def main() -> None:

    parser = ArgumentParser(description='Generate "lorem ipsum" style text with PyDTMC.')
    parser.add_argument('--corpus', type=Path, default=Path(__file__).with_name('corpus.txt'))
    parser.add_argument('--tokens', type=int, default=80, help='The number of tokens to generate.')
    parser.add_argument('--seed', type=int, default=42, help='The random seed.')
    args = parser.parse_args()

    if args.tokens < 4:
        parser.error('tokens must be greater than or equal to four.')

    generator = TextGenerator.from_file(args.corpus)
    output = generator.generate(tokens=args.tokens, seed=args.seed)

    print(f'Augmented states: {generator.states}')
    print(f'Output: "{output}"')


if __name__ == '__main__':
    main()
