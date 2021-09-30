# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from os.path import (
    abspath as _os_abspath,
    dirname as _os_dirname,
    isfile as _os_isfile,
    join as _os_join
)

from json import (
    load as _json_load
)


#############
# CONSTANTS #
#############

_replacements = [
    ('NaN', float('nan')),
    ('-Infinity', float('-inf')),
    ('Infinity', float('inf'))
]


###########
# CACHING #
###########

_fixtures = {}


#############
# FUNCTIONS #
#############

def _sanitize_fixture_recursive(element, replacements):

    if isinstance(element, dict):
        return {key: _sanitize_fixture_recursive(value, replacements) for key, value in element.items()}

    if isinstance(element, list):
        return [_sanitize_fixture_recursive(item, replacements) for item in element]

    for replacement in replacements:
        if element == replacement[0]:
            return replacement[1]

    return element


def _parse_fixture_dictionary(fixture, fixture_names, subtest_name):

    values = []
    ids = []

    expected_args = len(fixture_names)
    subtest_reference = f'{subtest_name.replace("test_", "")}_data'

    if subtest_reference in fixture:

        fixture_data = fixture[subtest_reference]

        if isinstance(fixture_data, dict):

            values_current = tuple(fixture_data[fixture_name] for fixture_name in fixture_names if fixture_name in fixture_data)

            if len(values_current) == expected_args:
                values.append(values_current)
                ids.append(f'{subtest_name}')

        elif isinstance(fixture_data, list):

            for index, case in enumerate(fixture_data):

                case_id = f'_{case["id"]}' if 'id' in case else f' #{str(index + 1)}'
                values_current = tuple(case[fixture_name] for fixture_name in fixture_names if fixture_name in case)

                if len(values_current) == expected_args:
                    values.append(values_current)
                    ids.append(f'{subtest_name}{case_id}')

            if len(values) != len(fixture_data):
                values = []
                ids = []

    return values, ids


def _parse_fixture_list(fixture, fixture_names, subtest_name):

    values = []
    ids = []

    expected_args = len(fixture_names)
    subtest_reference = f'{subtest_name.replace("test_", "")}_data'

    if any(subtest_reference in case for case in fixture):

        flags = [False] * len(fixture)

        for index_case, case in enumerate(fixture):

            if subtest_reference in case:

                case_id = case['id'] if 'id' in case else f' #{str(index_case + 1)}'
                case_values = tuple(case[fixture_name] for fixture_name in fixture_names if fixture_name in case)

                for index_subcase, subcase in enumerate(case[subtest_reference]):

                    values_current = case_values + tuple(subcase[fixture_name] for fixture_name in fixture_names if fixture_name in subcase)

                    if len(values_current) == expected_args:

                        values.append(values_current)
                        ids.append(f'{subtest_name} {case_id}-{str(index_subcase + 1)}')

                        flags[index_case] = True

        if not all(flags):
            values = []
            ids = []

    else:

        for index, case in enumerate(fixture):

            case_id = case['id'] if 'id' in case else f' #{str(index + 1)}'
            values_current = tuple(case[fixture_name] for fixture_name in fixture_names if fixture_name in case)

            if len(values_current) == expected_args:
                values.append(values_current)
                ids.append(f'{subtest_name} {case_id}')

        if len(values) != len(fixture):
            values = []
            ids = []

    return values, ids


#########
# SETUP #
#########

def pytest_configure(config):

    config.addinivalue_line('filterwarnings', 'ignore::DeprecationWarning')
    config.addinivalue_line('filterwarnings', 'ignore::PendingDeprecationWarning')
    config.addinivalue_line('filterwarnings', 'ignore::matplotlib.cbook.mplDeprecation')

    config.addinivalue_line('markers', 'slow: mark tests as slow (exclude them with \'-m "not slow"\').')


def pytest_generate_tests(metafunc):

    module = metafunc.module.__name__
    func = metafunc.definition.name
    mark = metafunc.definition.get_closest_marker('parametrize')
    names = metafunc.fixturenames

    test_index = module.find('_') + 1
    test_name = module[test_index:]

    if test_name not in _fixtures:

        base_directory = _os_abspath(_os_dirname(__file__))
        fixtures_file = _os_join(base_directory, f'fixtures/fixtures_{test_name}.json')

        if not _os_isfile(fixtures_file):
            _fixtures[test_name] = None
        else:

            with open(fixtures_file, 'r') as file:
                fixture = _json_load(file)
                fixture = _sanitize_fixture_recursive(fixture, _replacements)
                _fixtures[test_name] = fixture

    fixture = _fixtures[test_name]

    values = []
    ids = []

    if len(names) > 0 and mark is None and fixture is not None and len(fixture) > 0:

        if isinstance(fixture, dict):
            values, ids = _parse_fixture_dictionary(fixture, names, func)
        elif isinstance(fixture, list):
            values, ids = _parse_fixture_list(fixture, names, func)

    metafunc.parametrize(names, values, False, ids)
