from csv import DictReader
from pathlib import Path

import numpy as np

from pydtmc import MarkovChain


class SubscriptionModel:  # pylint: disable=missing-class-docstring

    _states = ['active', 'at_risk', 'dormant', 'churned']
    _contributions = {'active': 20.0, 'at_risk': 12.0, 'dormant': 0.0, 'churned': 0.0}

    def __init__(self, customers: dict[str, list[str]]):

        customers = customers.copy()

        counts = np.zeros((len(self._states), len(self._states)), dtype=float)
        indices = {state: index for index, state in enumerate(self._states)}

        for sequence in customers.values():
            for source, target in zip(sequence, sequence[1:]):
                counts[indices[source], indices[target]] += 1.0

        p = np.zeros_like(counts, dtype=float)

        for index, state in enumerate(self._states):

            if state == 'churned':
                p[index, index] = 1.0
            else:

                total = np.sum(counts[index])

                if total == 0:
                    raise ValueError(f'No observed transitions from {state}.')

                p[index] = counts[index] / total

        self._chain = MarkovChain(p, self._states)
        self._customers = customers

    @property
    def contributions(self) -> dict[str, float]:
        return self._contributions.copy()

    @property
    def customers(self) -> dict[str, list[str]]:
        return self._customers.copy()

    @property
    def customers_count(self) -> int:
        return len(self._customers)

    @property
    def states(self) -> list[str]:
        return self._states.copy()

    @property
    def states_count(self) -> int:
        return len(self._states)

    @property
    def transition_matrix(self) -> np.ndarray:
        return self._chain.p.copy()

    def churn_probability(self, months: int, state: str = 'active') -> float:

        if months < 0:
            raise ValueError('Months must be non-negative.')

        x = np.linalg.matrix_power(self._chain.p, months)

        index = self._states.index(state)
        terminal = self._states.index('churned')
        value = float(x[index, terminal])

        return value

    def discounted_contribution(self, discount_rate: float = 0.01, state: str = 'active') -> float:

        if discount_rate < 0.0:
            raise ValueError('Discount rate must be non-negative.')

        beta = 1.0 / (1.0 + discount_rate)

        transient = self._states[:-1]
        q = self._chain.p[:-1, :-1]
        rewards = np.array([self._contributions[name] for name in transient], dtype=float)

        value = float(np.linalg.solve(np.eye(len(transient)) - beta * q, rewards)[transient.index(state)])

        return value

    def expected_months_to_churn(self, state: str = 'active') -> float:

        transient = self._states[:-1]
        q = self._chain.p[:-1, :-1]
        times = np.linalg.solve(np.eye(len(transient)) - q, np.ones(len(transient)))

        value = float(times[transient.index(state)])

        return value

    def observed_lifetime(self) -> float:

        value = sum(len(sequence) - 1 for sequence in self.customers.values()) / len(self.customers)

        return value

    def observed_transitions(self) -> int:

        value = sum(len(sequence) - 1 for sequence in self.customers.values())

        return value

    @classmethod
    def from_csv(cls, path: str | Path) -> 'SubscriptionModel':

        path = path if isinstance(path, str) else Path(path)

        customers = {}

        with path.open('r', newline='', encoding='utf-8') as file:

            reader = DictReader(file)

            if reader.fieldnames != ['customer_id', 'month', 'state']:
                raise ValueError('Expected CSV columns are: customer_id, month, state.')

            for row in reader:

                customer_id = row['customer_id']
                month_text = row['month']
                state = row['state']

                if (not customer_id) or (not month_text) or (not month_text.isdecimal()):
                    raise ValueError('Customer IDs and non-negative integer months are required.')

                if state not in cls._states:
                    raise ValueError(f'Unknown subscription state detected: {state!r}.')

                month = int(month_text)
                sequence = customers.setdefault(customer_id, [])

                if month != len(sequence):
                    raise ValueError(f'Invalid month ordering for customer {customer_id}.')

                if sequence and (sequence[-1] == 'churned'):
                    raise ValueError(f'Observations after churn detected for customer {customer_id}.')

                if not sequence and (state != 'active'):
                    raise ValueError(f'Customer {customer_id} must be active.')

                sequence.append(state)

        if not customers:
            raise ValueError('No customer histories provided.')

        if any(sequence[-1] != 'churned' for sequence in customers.values()):
            raise ValueError('All histories must terminate at churn.')

        instance = cls(customers)

        return instance
