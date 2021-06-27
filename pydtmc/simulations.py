# -*- coding: utf-8 -*-

__all__ = [
    'predict',
    'redistribute',
    'simulate',
    'walk_probability'
]


###########
# IMPORTS #
###########

# Libraries

import numpy as np

# Internal

from .custom_types import (
    oint,
    olist_int,
    tarray,
    tlist_int,
    tmc,
    trand,
    tredists
)


#############
# FUNCTIONS #
#############

def predict(mc: tmc, steps: int, initial_state: int) -> olist_int:

    current_state = initial_state
    value = [initial_state]

    for _ in range(steps):

        d = mc.p[current_state, :]
        d_max = np.argwhere(d == np.max(d))

        if d_max.size > 1:
            return None

        current_state = d_max.item()
        value.append(current_state)

    return value


def redistribute(mc: tmc, steps: int, initial_status: tarray, output_last: bool) -> tredists:

    value = np.zeros((steps + 1, mc.size), dtype=float)
    value[0, :] = initial_status

    for i in range(1, steps + 1):
        value[i, :] = value[i - 1, :].dot(mc.p)
        value[i, :] /= np.sum(value[i, :])

    if output_last:
        return value[-1]

    value = [np.ravel(distribution) for distribution in np.split(value, value.shape[0])]

    return value


def simulate(mc: tmc, steps: int, initial_state: int, final_state: oint, rng: trand) -> tlist_int:

    current_state = initial_state
    value = [initial_state]

    for _ in range(steps):

        w = mc.p[current_state, :]
        current_state = rng.choice(mc.size, size=1, p=w).item()
        value.append(current_state)

        if final_state is not None and current_state == final_state:
            break

    return value


def walk_probability(mc: tmc, walk: tlist_int) -> float:

    p = 0.0

    for (i, j) in zip(walk[:-1], walk[1:]):

        if mc.p[i, j] > 0.0:
            p += np.log(mc.p[i, j])
        else:
            p = -np.inf
            break

    value = np.exp(p)

    return value
