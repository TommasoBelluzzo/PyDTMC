# -*- coding: utf-8 -*-

__all__ = [
    'read_csv',
    'read_json',
    'read_txt',
    'write_csv',
    'write_json',
    'write_txt'
]


###########
# IMPORTS #
###########

# Full

import numpy as np

# Partial

from csv import (
    reader as csv_reader,
    writer as csv_writer,
    QUOTE_MINIMAL as csv_quote_minimal
)

from json import (
    dump as json_dump,
    load as json_load
)

# Internal

from .custom_types import *


#############
# FUNCTIONS #
#############

def read_csv(file_path: str) -> tmc_dict:

    d = {}

    size = 0
    states = None

    with open(file_path, mode='r', newline='') as file:

        data = csv_reader(file)

        for index, row in enumerate(data):

            if index == 0:

                states = row

                if not all(isinstance(s, str) for s in states) or not all(s is not None and len(s) > 0 for s in states):
                    raise ValueError('The file header is invalid.')

                size = len(states)
                states_unique = len(set(states))

                if states_unique < size:
                    raise ValueError('The file header is invalid.')

            else:

                probabilities = row

                if len(probabilities) != size or not all(isinstance(p, str) for p in probabilities) or not all(p is not None and len(p) > 0 for p in probabilities):
                    raise ValueError('The file contains invalid lines.')

                state_from = states[index - 1]

                for i in range(size):

                    state_to = states[i]

                    try:
                        probability = float(probabilities[i])
                    except Exception:
                        raise ValueError('The file contains invalid lines.')

                    d[(state_from, state_to)] = probability

    return d


def read_json(file_path: str) -> tmc_dict:

    d = {}

    with open(file_path, mode='r') as file:

        file.seek(0)

        if not file.read(1):
            raise OSError('The file is empty.')
        else:
            file.seek(0)

        data = json_load(file)

        if not isinstance(data, list):
            raise ValueError('The file is malformed.')

        for obj in data:

            if not isinstance(obj, dict):
                raise ValueError('The file contains invalid entries.')

            if sorted(list(set(obj.keys()))) != ['probability', 'state_from', 'state_to']:
                raise ValueError('The file contains invalid lines.')

            state_from = obj['state_from']
            state_to = obj['state_to']
            probability = obj['probability']

            if not isinstance(state_from, str) or not isinstance(state_to, str) or not isinstance(probability, (float, int)):
                raise ValueError('The file contains invalid lines.')

            d[(state_from, state_to)] = float(probability)

    return d


def read_txt(file_path: str) -> tmc_dict:

    d = {}

    with open(file_path, mode='r') as file:

        file.seek(0)

        if not file.read(1):
            raise OSError('The file is empty.')
        else:
            file.seek(0)

        for line in file:

            if not line.strip():
                raise ValueError('The file contains invalid lines.')

            ls = line.split()

            if len(ls) != 3:
                raise ValueError('The file contains invalid lines.')

            try:
                ls2 = float(ls[2])
            except Exception:
                raise ValueError('The file contains invalid lines.')

            d[(ls[0], ls[1])] = ls2

    return d


def write_csv(d: tmc_dict, file_path: str):

    states = [key[0] for key in d.keys() if key[0] == key[1]]
    size = len(states)

    p = np.zeros((size, size), dtype=float)

    for it, ip in d.items():
        p[states.index(it[0]), states.index(it[1])] = ip

    with open(file_path, mode='w', newline='') as file:

        writer = csv_writer(file, delimiter=',', quoting=csv_quote_minimal, quotechar='"')

        writer.writerow(states)

        for i in range(size):
            row = [str(x) for x in p[i, :].tolist()]
            writer.writerow(row)


def write_json(d: tmc_dict, file_path: str):

    output = []

    for it, ip in d.items():
        output.append({'state_from': it[0], 'state_to': it[1], 'probability': ip})

    with open(file_path, mode='w') as file:
        json_dump(output, file)


def write_txt(d: tmc_dict, file_path: str):

    with open(file_path, mode='w') as file:

        for it, ip in d.items():
            file.write(f'{it[0]} {it[1]} {ip}\n')
