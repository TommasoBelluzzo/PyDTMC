# -*- coding: utf-8 -*-

__all__ = [
    'create_rng',
    'generate_validation_error',
    'get_file_extension'
]


###########
# IMPORTS #
###########

# Full

import numpy.random as npr
import numpy.random.mtrand as nprm
import pathlib as pl

# Internal

from .custom_types import *
from .exceptions import *


#############
# FUNCTIONS #
#############

# noinspection PyProtectedMember
def create_rng(seed: oint) -> trand:

    if seed is None:
        return nprm._rand

    if isinstance(seed, int):
        return npr.RandomState(seed)

    raise TypeError('The specified seed is not a valid RNG initializer.')


def generate_validation_error(e: texception, trace: tany) -> ValidationError:

    arguments = ''.join(trace[0][4]).split('=', 1)[0].strip()
    message = str(e).replace('@arg@', arguments)

    return ValidationError(message)


def get_file_extension(file_path: str) -> str:

    return ''.join(pl.Path(file_path).suffixes).lower()


def namedtuple_to_dictionary(obj: tany):

    if isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # noinspection PyProtectedMember
        return dict(zip(obj._fields, map(namedtuple_to_dictionary, obj)))

    if isinstance(obj, titerable) and not isinstance(obj, str):
        return type(obj)(map(namedtuple_to_dictionary, obj))

    if isinstance(obj, tmapping):
        return type(obj)(zip(obj.keys(), map(namedtuple_to_dictionary, obj.values())))

    return obj


INDENT = 2
SPACE = " "
NEWLINE = "\n"

from math import (
    isnan,
    isinf
)


# Changed basestring to str, and dict uses items() instead of iteritems().
def to_json(o, level=0, force=False):
  ret = ""

  if isinstance(o, dict):
    if force:
        ret += NEWLINE
    ret += SPACE * INDENT * level + "{" + NEWLINE
    comma = ''
    for k, v in o.items():
      if k == '_type':
          continue
      ret += comma
      comma = ",\n"
      ret += SPACE * INDENT * (level + 1)
      ret += '"' + str(k) + '":' + SPACE
      ret += to_json(v, level + 1)
    ret += NEWLINE + SPACE * INDENT * level + "}"
  elif isinstance(o, str):
    ret += '"' + o + '"'
  elif isinstance(o, list):
    if level == 0 or (len(o) > 0 and isinstance(o[0], dict)):
        ret += "[" + NEWLINE + ",".join([to_json(e, level + 1, True if idx > 0 else False) for idx, e in enumerate(o)]) + NEWLINE + SPACE * INDENT * level + "]"
    else:
        ret += "[" + ", ".join([to_json(e, level + 1) for e in o]) + "]"
  elif isinstance(o, bool):
    ret += "true" if o else "false"
  elif isinstance(o, int):
    ret += str(o)
  elif isinstance(o, float):
    if isnan(o):
        ret += '"NaN"'
    elif isinf(o):
        if o > 0.0:
            ret += '"Infinity"'
        else:
            ret += '"-Infinity"'
    else:
        ret += str(o)
  elif o is None:
    ret += 'null'
  else:
    raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
  return ret
