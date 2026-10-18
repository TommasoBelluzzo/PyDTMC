from argparse import ArgumentParser
from pathlib import Path
from model import ConversionModel


def main() -> None:

    parser = ArgumentParser(description='Analyze website conversion with PyDTMC.')
    parser.add_argument('--sessions', type=Path, default=Path(__file__).with_name('sessions.csv'))
    args = parser.parse_args()

    model = ConversionModel.from_csv(args.sessions)
    conversion = model.conversion_probability()
    abandonment = model.abandonment_probability()
    steps = model.expected_steps_to_outcome()

    assert abs(conversion + abandonment - 1.0) < 1e-10
    assert 0.0 <= conversion <= 1.0
    assert steps > 0.0

    print(f'Sessions: {model.sessions_count}')
    print(f'Observed conversion rate: {model.observed_conversion_rate():.2%}')
    print(f'Model conversion probability: {conversion:.2%}')
    print(f'Model abandonment probability: {abandonment:.2%}')
    print(f'Expected steps to outcome: {steps:.2f}')
    print('Conversion probability by page:')

    for state in model.states:
        if state not in (model.success_state, model.failure_state):
            print(f'  {state}: {model.conversion_probability(state):.2%}')


if __name__ == '__main__':
    main()
