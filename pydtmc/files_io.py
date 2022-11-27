# -*- coding: utf-8 -*-

__all__ = [
    'read_csv',
    'read_json',
    'read_txt',
    'read_xml',
    'write_csv',
    'write_json',
    'write_txt',
    'write_xml'
]


###########
# IMPORTS #
###########

# Standard

import csv as _csv
import io as _io
import json as _json
import xml.etree as _xmle
import xml.etree.ElementTree as _xmlet

try:
    from defusedxml.ElementTree import parse as _xml_parse
except ImportError:  # pragma: no cover
    from xml.etree.ElementTree import parse as _xml_parse

# Libraries

import numpy as _np

# Internal

from .custom_types import (
    tobj_dict as _tobj_dict,
    tpath as _tpath
)


#############
# CONSTANTS #
#############

_valid_params_hmm = {
    'reference': ('string', ('E', 'P')),
    'element_from': ('string', ()),
    'element_to': ('string', ()),
    'probability': ('number', ())
}

_valid_params_mc = {
    'state_from': ('string', ()),
    'state_to': ('string', ()),
    'probability': ('number', ())
}


#############
# FUNCTIONS #
#############

def read_csv(mc: bool, file_path: _tpath) -> _tobj_dict:

    def _read_csv_hmm(rch_data):

        labels, states, symbols = [], [], []
        n, k, nk = 0, 0, 0

        result = {}

        for index, row in enumerate(rch_data):

            if index == 0:

                for label in row:

                    if not isinstance(label, str) or len(label) < 3:  # pragma: no cover
                        raise ValueError('The file header is invalid.')

                    label = label.strip()
                    label_prefix = label[:2]

                    if label_prefix == 'E_':
                        label_value = label[2:]
                        symbols.append(label_value)
                        labels.append((label_prefix[:1], label_value))
                    elif label_prefix == 'P_':
                        label_value = label[2:]
                        states.append(label_value)
                        labels.append((label_prefix[:1], label_value))
                    else:  # pragma: no cover
                        raise ValueError('The file header is invalid.')

                n, k = len(states), len(symbols)

                if len(set(states)) < n or len(set(symbols)) < k:  # pragma: no cover
                    raise ValueError('The file header is invalid.')

                if n != len(rch_data) - 1:  # pragma: no cover
                    raise ValueError('The file contains an invalid number of rows.')

                nk = n + k

            else:

                if len(row) != nk or not all(isinstance(p, str) and len(p) > 0 for p in row):  # pragma: no cover
                    raise ValueError('The file contains invalid rows.')

                element_from = states[index - 1]

                for i in range(nk):

                    label_prefix, element_to = labels[i]

                    try:
                        probability = float(row[i])
                    except Exception as ex:  # pragma: no cover
                        raise ValueError('The file contains invalid rows.') from ex

                    result[(label_prefix, element_from, element_to)] = probability

        return result

    def _read_csv_mc(rcm_data):

        states = []
        n = 0

        result = {}

        for index, row in enumerate(rcm_data):

            if index == 0:

                if not all(isinstance(state, str) and len(state) > 0 for state in row):  # pragma: no cover
                    raise ValueError('The file header is invalid.')

                states = row
                n = len(states)

                if len(set(states)) < n:  # pragma: no cover
                    raise ValueError('The file header is invalid.')

                if n != len(rcm_data) - 1:  # pragma: no cover
                    raise ValueError('The file contains an invalid number of rows.')

            else:

                if len(row) != n or not all(isinstance(p, str) and len(p) > 0 for p in row):  # pragma: no cover
                    raise ValueError('The file contains invalid rows.')

                state_from = states[index - 1]

                for i in range(n):

                    state_to = states[i]

                    try:
                        probability = float(row[i])
                    except Exception as ex:  # pragma: no cover
                        raise ValueError('The file contains invalid rows.') from ex

                    result[(state_from, state_to)] = probability

        return result

    with open(file_path, mode='r', newline='') as file:
        file.seek(0)
        reader = _csv.reader(file)
        data = [row for _, row in enumerate(reader)]

    d = _read_csv_mc(data) if mc else _read_csv_hmm(data)

    return d


# noinspection PyTypeChecker
def read_json(mc: bool, file_path: _tpath) -> _tobj_dict:

    with open(file_path, mode='r') as file:
        file.seek(0)
        data = _json.load(file)

    if not isinstance(data, list):  # pragma: no cover
        raise ValueError('The file format is not compliant.')

    valid_params = _valid_params_mc if mc else _valid_params_hmm
    valid_params_keys = sorted(valid_params.keys())

    d = {}

    for obj in data:

        if not isinstance(obj, dict):  # pragma: no cover
            raise ValueError('The file format is not compliant.')

        if sorted(obj.keys()) != valid_params_keys:  # pragma: no cover
            raise ValueError('The file contains invalid elements.')

        values = []

        for param_name, (param_type, param_possible_values) in valid_params.items():

            value = obj[param_name]

            if param_type == 'number':

                if not isinstance(value, (float, int, _np.floating, _np.integer)):  # pragma: no cover
                    raise ValueError('The file contains invalid elements.')

                value = float(value)

            elif param_type == 'string':

                if not isinstance(value, str) or len(value) == 0:  # pragma: no cover
                    raise ValueError('The file contains invalid elements.')

            if len(param_possible_values) > 0 and value not in param_possible_values:  # pragma: no cover
                raise ValueError('The file contains invalid elements.')

            values.append(value)

        d[tuple(values[:-1])] = values[-1]

    return d


# noinspection PyTypeChecker
def read_txt(mc: bool, file_path: _tpath) -> _tobj_dict:

    with open(file_path, mode='r') as file:
        file.seek(0)
        data = [line.strip().split() for line in file]

    valid_params = _valid_params_mc if mc else _valid_params_hmm
    valid_params_keys = tuple(valid_params.keys())
    valid_params_length = len(valid_params)

    d = {}

    for row in data:

        if len(row) != valid_params_length:  # pragma: no cover
            raise ValueError('The file contains invalid lines.')

        values = []

        for param_name, (param_type, param_possible_values) in valid_params.items():

            value = row[valid_params_keys.index(param_name)]

            if param_type == 'number':
                try:
                    value = float(value)
                except Exception as ex:  # pragma: no cover
                    raise ValueError('The file contains invalid lines.') from ex

            if len(param_possible_values) > 0 and value not in param_possible_values:  # pragma: no cover
                raise ValueError('The file contains invalid lines.')

            values.append(value)

        d[tuple(values[:-1])] = values[-1]

    return d


# noinspection PyTypeChecker
def read_xml(mc: bool, file_path: _tpath) -> _tobj_dict:

    try:
        document = _xml_parse(file_path)
    except Exception as ex:  # pragma: no cover
        raise ValueError('The file format is not compliant.') from ex

    root = document.getroot()
    root_tag = 'MarkovChain' if mc else 'HiddenMarkovModel'

    if root.tag != root_tag:  # pragma: no cover
        raise ValueError('The file root element is invalid.')

    valid_params = _valid_params_mc if mc else _valid_params_hmm
    valid_params_keys = sorted(valid_params.keys())

    d = {}

    for element in root.iter():

        if element.tag == root_tag:
            continue

        if element.tag != 'Item':  # pragma: no cover
            raise ValueError('The file contains invalid elements.')

        attributes = element.items()

        if len(attributes) == 0:  # pragma: no cover
            raise ValueError('The file contains invalid elements.')

        attributes_keys = [attribute[0] for attribute in attributes]

        if sorted(attributes_keys) != valid_params_keys:  # pragma: no cover
            raise ValueError('The file contains invalid elements.')

        attributes_values = [attribute[1].strip() for attribute in attributes]

        if any(len(attributes_value) == 0 for attributes_value in attributes_values):  # pragma: no cover
            raise ValueError('The file contains invalid elements.')

        values = []

        for param_name, (param_type, param_possible_values) in valid_params.items():

            value = attributes_values[attributes_keys.index(param_name)]

            if param_type == 'number':
                try:
                    value = float(value)
                except Exception as ex:  # pragma: no cover
                    raise ValueError('The file contains invalid elements.') from ex

            if len(param_possible_values) > 0 and value not in param_possible_values:  # pragma: no cover
                raise ValueError('The file contains invalid elements.')

            values.append(value)

        d[tuple(values[:-1])] = values[-1]

    return d


def write_csv(mc: bool, d: _tobj_dict, file_path: _tpath):

    def _write_csv_hmm(wch_d):

        states = [key[1] for key in wch_d.keys() if key[0] == 'P' and key[1] == key[2]]
        symbols = [key[2] for key in wch_d.keys() if key[0] == 'E' and key[1] == states[0]]
        n, k = len(states), len(symbols)
        p, e = _np.zeros((n, n), dtype=float), _np.zeros((n, k), dtype=float)

        for (reference, element_from, element_to), probability in d.items():
            if reference == 'E':
                e[states.index(element_from), symbols.index(element_to)] = probability
            else:
                p[states.index(element_from), states.index(element_to)] = probability

        header = [f'P_{state}' for state in states] + [f'E_{symbol}' for symbol in symbols]
        rows = []

        for i in range(n):
            rows.append([str(x) for x in p[i, :].tolist()] + [str(x) for x in e[i, :].tolist()])

        return header, rows

    def _write_csv_mc(wcm_d):

        states = [key[0] for key in wcm_d.keys() if key[0] == key[1]]
        size = len(states)

        p = _np.zeros((size, size), dtype=float)

        for (state_from, state_to), probability in d.items():
            p[states.index(state_from), states.index(state_to)] = probability

        header = states
        rows = []

        for i in range(size):
            rows.append([str(x) for x in p[i, :].tolist()])

        return header, rows

    header_out, rows_out = _write_csv_mc(d) if mc else _write_csv_hmm(d)

    with open(file_path, mode='w', newline='') as file:

        writer = _csv.writer(file, delimiter=',', quoting=_csv.QUOTE_MINIMAL, quotechar='"')

        writer.writerow(header_out)

        for row in rows_out:
            writer.writerow(row)


def write_json(mc: bool, d: _tobj_dict, file_path: _tpath):

    valid_params_keys = tuple((_valid_params_mc if mc else _valid_params_hmm).keys())

    data = []

    for key, value in d.items():

        item = {}

        for index, attribute in enumerate(key):
            item[valid_params_keys[index]] = attribute

        item[valid_params_keys[-1]] = value

        data.append(item)

    with open(file_path, mode='w') as file:
        _json.dump(data, file)


def write_txt(d: _tobj_dict, file_path: _tpath):

    with open(file_path, mode='w') as file:

        line = ''

        for key, value in d.items():

            for attribute in key:
                line += f'{attribute} '

            line += f'{value}\n'

        file.write(line)


def write_xml(mc: bool, d: _tobj_dict, file_path: _tpath):

    valid_params_keys = tuple((_valid_params_mc if mc else _valid_params_hmm).keys())
    root_tag = 'MarkovChain' if mc else 'HiddenMarkovModel'

    root = _xmlet.Element(root_tag)

    for key, value in d.items():

        item = _xmlet.SubElement(root, 'Item')

        for index, attribute in enumerate(key):
            item.set(valid_params_keys[index], attribute)

        item.set(valid_params_keys[-1], str(value))

    document = _xmle.ElementTree.ElementTree(root)

    with _io.BytesIO() as buffer:
        document.write(buffer, 'utf-8', True)
        xml_content = str(buffer.getvalue(), 'utf-8')

    xml_content = xml_content.replace('?>', " standalone='yes' ?>")
    xml_content = xml_content.replace(f'<{root_tag}>', f'<{root_tag}>\n')
    xml_content = xml_content.replace('<Item', '\t<Item')
    xml_content = xml_content.replace('" />', '"/>\n')

    with open(file_path, mode='w') as file:
        file.write(xml_content)
