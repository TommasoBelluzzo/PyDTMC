from argparse import ArgumentParser
from pathlib import Path

from model import ReliabilityModel  # pylint: disable=import-error


def main() -> None:

    parser = ArgumentParser(description='Analyze server availability with PyDTMC.')
    parser.add_argument('--incidents', type=Path, default=Path(__file__).with_name('incidents.csv'))
    parser.add_argument('--hours', type=int, default=24, help='The forecast horizon in hourly observations.')
    args = parser.parse_args()

    model = ReliabilityModel.from_csv(args.incidents)

    print(f'Observations: {len(model.incidents)}')
    print(f'First timestamp: {model.incidents[0][0]:%Y-%m-%d %H:%M}')
    print(f'Last timestamp: {model.incidents[-1][0]:%Y-%m-%d %H:%M}')
    print('')
    print(f'Long-run availability: {model.availability():.2%}')
    print(f'Long-run failed fraction: {model.failed_fraction():.2%}')
    print(f'Mean failure-state recurrence: {model.mean_failure_recurrence():.2f}')
    print(f'Expected available hours: {model.expected_available_hours(args.hours):.2f}')
    print('')
    print('Stationary distribution:')

    for state, probability in model.stationary_distribution.items():
        print(f'  {state}: {probability:.2%}')


if __name__ == '__main__':
    main()
