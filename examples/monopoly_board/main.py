from argparse import ArgumentParser
from pathlib import Path

from model import MonopolyModel  # pylint: disable=import-error


def main() -> None:

    parser = ArgumentParser(description='Analyze Monopoly board positions with PyDTMC.')
    parser.add_argument('--board', type=Path, default=Path(__file__).with_name('board.json'))
    parser.add_argument('--rolls', type=int, default=100, help='The number of rolls for expected visits.')
    parser.add_argument('--seed', type=int, default=42, help='The random seed.')
    args = parser.parse_args()

    if args.rolls < 2:
        parser.error('rolls must be greater than or equal to 2.')

    model = MonopolyModel.from_json(args.board)
    top_squares = model.top_squares(10)
    expected_visits = model.expected_visits(args.rolls)
    simulation = model.simulate(args.rolls, args.seed)

    assert abs(float(expected_visits.sum()) - args.rolls) < 1e-8

    print(f'Squares: {len(model.squares)}')
    print(f'Rolls: {args.rolls}')
    print('')
    print('Top 10 squares by stationary probability:')

    for position, name, probability in top_squares:
        print(f'  {position + 1:02d}) {name}: {probability:.2%}')

    print('')
    print('Top 10 squares by expected visits:')

    for index in (-expected_visits).argsort(kind='stable')[:10]:
        print(f'  {index + 1:02d}) {model.squares[index]}: {expected_visits[index]:.2f}')

    print('')
    print('First 10 simulated positions:')

    for position, name in simulation[:10]:
        print(f'  {position + 1:02d}) {name}')


if __name__ == '__main__':
    main()
