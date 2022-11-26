# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Libraries

import pytest as _pt


#############
# CONSTANTS #
#############

_benchmark_options = {
    'warmup_rounds': 3,
    'rounds': 10,
    'iterations': 100
}


##############
# BENCHMARKS #
##############


@_pt.mark.benchmark
def test_benchmark_generators(benchmark, func_name, func_obj, func_args):

    benchmark.group = func_name
    benchmark.pedantic(func_obj, kwargs=func_args, **_benchmark_options)

    assert True


@_pt.mark.benchmark
def test_benchmark_measures(benchmark, func_name, func_obj, func_args):

    benchmark.group = func_name
    benchmark.pedantic(func_obj, kwargs=func_args, **_benchmark_options)

    assert True
