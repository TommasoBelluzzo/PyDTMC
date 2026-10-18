from argparse import ArgumentParser
from pathlib import Path

from model import SubscriptionModel  # pylint: disable=import-error


def main() -> None:

    parser = ArgumentParser(description='Analyze subscription dynamics with PyDTMC.')
    parser.add_argument('--customers', type=Path, default=Path(__file__).with_name('customers.csv'))
    parser.add_argument('--months', type=int, default=12, help='The retention horizon in months.')
    parser.add_argument('--discount_rate', type=float, default=0.01, help='The monthly discount rate.')
    args = parser.parse_args()

    if args.months < 1:
        parser.error('months must be positive.')

    if args.discount_rate < 0.0:
        parser.error('discount rate must be non-negative.')

    model = SubscriptionModel.from_csv(args.customers)
    observed_transitions = model.observed_transitions()
    observed_lifetime = model.observed_lifetime()
    churn_probability = model.churn_probability(args.months)
    expected_months_to_churn = model.expected_months_to_churn()
    discounted_contribution = model.discounted_contribution(args.discount_rate)

    assert 0.0 <= churn_probability <= 1.0
    assert expected_months_to_churn > 0.0
    assert discounted_contribution >= 0.0

    print(f'Customers: {model.customers_count}')
    print(f'Observed transitions: {observed_transitions}')
    print(f'Observed mean lifetime: {observed_lifetime:.2f} months')
    print('')
    print('Transition matrix:')
    print(model.transition_matrix)
    print(f'Mean time to churn: {expected_months_to_churn:.2f} months')
    print(f'Discounted lifetime contribution: {discounted_contribution:.2f} units')
    print('')
    print(f'Churn probability: {churn_probability:.2%}')
    print('Churn by starting state:')

    for state in model.states[:-1]:
        print(f'  {state}: {model.churn_probability(args.months, state):.2%}')

    print(f'Retention probability: {1.0 - churn_probability:.2%}')
    print('Retention by starting state:')

    for state in model.states[:-1]:
        print(f'  {state}: {1.0 - model.churn_probability(args.months, state):.2%}')


if __name__ == '__main__':
    main()
