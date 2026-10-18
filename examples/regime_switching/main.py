from argparse import ArgumentParser
from pathlib import Path

from model import RegimeModel  # pylint: disable=import-error


def main() -> None:

    parser = ArgumentParser(description='Identify synthetic market regimes with PyDTMC.')
    parser.add_argument('--observations', type=Path, default=Path(__file__).with_name('observations.csv'))
    args = parser.parse_args()

    model = RegimeModel.from_csv(args.observations)
    model.fit()

    print(f'Observations: {model.observations_count}')
    print(f'First date: {model.observations[0][0]:%Y-%m-%d}')
    print(f'Last date: {model.observations[-1][0]:%Y-%m-%d}')
    print('')
    print(f'Observation symbols: {model.symbols_count}')
    print(f'Hidden regimes: {model.states_count}')
    print(f'Inferred regime changes: {model.changes}')
    print('')
    print('Transition matrix:')
    print(model.transition_matrix)
    print('Emission matrix:')
    print(model.emission_matrix)
    print('')
    print('Last 10 inferred regimes:')

    for (day, value), regime in zip(model.observations[-10:], model.regimes[-10:]):
        print(f'  {day:%Y-%m-%d}: {value:+.4%} ({regime})')


if __name__ == '__main__':
    main()
