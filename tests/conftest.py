# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import importlib as _il
import json as _json
import os.path as _osp

# Libraries

import numpy as _np


#############
# CONSTANTS #
#############

_base_directory = _osp.abspath(_osp.dirname(__file__))
_benchmark_exclusions = ('benchmark', 'request')
_numpy_formatting_options = _np.get_printoptions()
_json_replacements = (
    ('NaN', float('nan')),
    ('-Infinity', float('-inf')),
    ('Infinity', float('inf'))
)


###########
# CACHING #
###########

_fixtures = {}


#############
# FUNCTIONS #
#############

def _extract_fixtures(fixtures_file):

    fixtures_path = _osp.join(_base_directory, f'fixtures/fixtures_{fixtures_file}.json')

    if not _osp.isfile(fixtures_path):
        return None

    with open(fixtures_path, 'r') as file:
        fixtures = _json.load(file)

    fixtures = _sanitize_fixtures_recursive(fixtures)

    return fixtures


# noinspection PyBroadException
def _parse_fixtures_benchmark(fixtures, func):

    values, ids = [], []

    try:
        module_name = func.split('_')[2]
        module = _il.import_module(f'pydtmc.{module_name}')
    except Exception:
        return values, ids

    if module_name not in fixtures:
        return values, ids

    module_fixtures = fixtures[module_name]

    for func_name in dir(module):
        if not func_name.startswith('_') and func_name in module_fixtures:

            func = getattr(module, func_name)
            func_args = module_fixtures[func_name]

            values.append((func_name, func, func_args))
            ids.append(func_name)

    return values, ids


def _parse_fixtures_dictionary(fixtures, names, func):

    values, ids = [], []

    expected_args = len(names)
    target = f'{func.replace("test_", "")}_data'

    if target in fixtures:

        fixture = fixtures[target]

        if isinstance(fixture, dict):

            fixture_values = tuple(fixture[name] for name in names if name in fixture)

            if len(fixture_values) == expected_args:
                values.append(fixture_values)
                ids.append(f'{func}')

        elif isinstance(fixture, list):

            for case_index, case in enumerate(fixture):

                case_id = f'_{case["id"]}' if 'id' in case else f' #{str(case_index + 1)}'
                case_values = tuple(case[name] for name in names if name in case)

                if len(case_values) == expected_args:
                    values.append(case_values)
                    ids.append(f'{func}{case_id}')

            if len(values) != len(fixture):
                values = []
                ids = []

    return values, ids


def _parse_fixtures_list(fixtures, names, func):

    values, ids = [], []

    expected_args = len(names)
    target = f'{func.replace("test_", "")}_data'

    if any(target in fixture for fixture in fixtures):

        flags = [False] * len(fixtures)

        for fixture_index, fixture in enumerate(fixtures):

            if target in fixture:

                fixture_id = fixture['id'] if 'id' in fixture else f' #{str(fixture_index + 1)}'
                fixture_values = tuple(fixture[name] for name in names if name in fixture)

                for case_index, case in enumerate(fixture[target]):

                    case_id = f'{str(case_index + 1)}'
                    case_values = fixture_values + tuple(case[name] for name in names if name in case)

                    if len(case_values) == expected_args:
                        values.append(case_values)
                        ids.append(f'{func} {fixture_id}-{case_id}')
                        flags[fixture_index] = True

        if not all(flags):
            values = []
            ids = []

    else:

        for fixture_index, fixture in enumerate(fixtures):

            fixture_id = fixture['id'] if 'id' in fixture else f' #{str(fixture_index + 1)}'
            fixture_values = tuple(fixture[name] for name in names if name in fixture)

            if len(fixture_values) == expected_args:
                values.append(fixture_values)
                ids.append(f'{func} {fixture_id}')

        if len(values) != len(fixtures):
            values = []
            ids = []

    return values, ids


def _sanitize_fixtures_recursive(element):

    if isinstance(element, dict):
        return {key: _sanitize_fixtures_recursive(value) for key, value in element.items()}

    if isinstance(element, list):
        return [_sanitize_fixtures_recursive(item) for item in element]

    for replacement in _json_replacements:
        if element == replacement[0]:
            return replacement[1]

    return element


#########
# HOOKS #
#########

def pytest_configure(config):

    config.addinivalue_line('filterwarnings', 'ignore::DeprecationWarning')
    config.addinivalue_line('filterwarnings', 'ignore::PendingDeprecationWarning')
    config.addinivalue_line('filterwarnings', 'ignore::matplotlib.cbook.mplDeprecation')
    config.addinivalue_line('filterwarnings', 'ignore::numpy.VisibleDeprecationWarning')

    config.addinivalue_line('markers', 'slow:  mark tests as slow (exclude them with \'-m "not slow"\').')

    _np.set_printoptions(floatmode='fixed', precision=8)


def pytest_generate_tests(metafunc):

    names = metafunc.fixturenames

    if len(names) == 0:
        return

    mark = metafunc.definition.get_closest_marker('parametrize')

    if mark is not None:
        return

    module = metafunc.module.__name__
    func = metafunc.definition.name

    reference = module.split('.')[1]

    if reference not in _fixtures:
        fixtures_file = 'benchmarks' if reference == 'benchmarks' else reference[(reference.find('_') + 1):]
        _fixtures[reference] = _extract_fixtures(fixtures_file)

    fixtures = _fixtures[reference]
    values, ids = [], []

    if fixtures is not None and len(fixtures) > 0:
        if reference == 'benchmarks':
            names = [name for name in names if name not in _benchmark_exclusions]
            values, ids = _parse_fixtures_benchmark(fixtures, func)
        elif isinstance(fixtures, dict):
            values, ids = _parse_fixtures_dictionary(fixtures, names, func)
        elif isinstance(fixtures, list):
            values, ids = _parse_fixtures_list(fixtures, names, func)

    metafunc.parametrize(names, values, False, ids)


def pytest_unconfigure():

    if 'floatmode' in _numpy_formatting_options:
        _np.set_printoptions(floatmode=_numpy_formatting_options['floatmode'])

    if 'precision' in _numpy_formatting_options:
        _np.set_printoptions(precision=_numpy_formatting_options['precision'])
