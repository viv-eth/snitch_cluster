# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

import struct
from datetime import datetime
import numpy as np
import torch
import argparse
import json5
import pathlib
import humanize
import pyflexfloat as ff

# Maximum available size in TCDM (in bytes)
TCDM_HEAP_SIZE = 112 * 1024


def emit_license():
    s = (f"// Copyright {datetime.now().year} ETH Zurich and University of Bologna.\n"
         f"// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         f"// SPDX-License-Identifier: Apache-2.0\n\n")
    return s


def variable_attributes(alignment=None, section=None):
    attributes = ''
    if alignment:
        attributes = f'__attribute__ ((aligned ({alignment})))'
    if section:
        attributes += f' __attribute__ ((section ("{section}")))'
    return attributes


def format_vector_definition(type, uid, vector, alignment=None, section=None):
    attributes = variable_attributes(alignment, section)
    s = f'{type} {uid}[{len(vector)}] {attributes} = ' + '{\n'
    for el in vector:
        if type != 'char':
            el_str = f'{el}'
        else:
            el_str = f'0x{el:02x}'
        s += f'\t{el_str},\n'
    s += '};'
    return s


def format_vector_declaration(type, uid, vector, alignment=None, section=None):
    attributes = variable_attributes(alignment, section)
    s = f'{type} {uid}[{len(vector)}] {attributes};'
    return s


def format_scalar_definition(type, uid, scalar):
    s = f'{type} {uid} = {scalar};'
    return s


def format_ifdef_wrapper(macro, body):
    s = f'#ifdef {macro}\n'
    s += f'{body}\n'
    s += f'#endif // {macro}\n'
    return s


# bytearray assumed little-endian
def bytes_to_doubles(byte_array):
    double_size = struct.calcsize('d')  # Size of a double in bytes
    num_doubles = len(byte_array) // double_size

    # Unpack the byte array into a list of doubles
    doubles = []
    for i in range(num_doubles):
        double_bytes = byte_array[i * double_size:(i + 1) * double_size]
        double = struct.unpack('<d', double_bytes)[0]
        doubles.append(double)
    return doubles


def bytes_to_uint32s(byte_array):
    uint32_size = struct.calcsize('I')  # Size of a uint32 in bytes
    num_uints = len(byte_array) // uint32_size

    # Unpack the byte array into a list of uints
    uints = []
    for i in range(num_uints):
        uint32_bytes = byte_array[i * uint32_size:(i + 1) * uint32_size]
        uint = struct.unpack('<I', uint32_bytes)[0]
        uints.append(uint)
    return uints

def _variable_attributes(alignment=None, section=None):
    attributes = ''
    if alignment:
        attributes = f'__attribute__ ((aligned ({alignment})))'
    if section:
        attributes += f' __attribute__ ((section ("{section}")))'
    return attributes

def _alias_dtype(dtype):
    if dtype == '__fp8':
        return 'char'
    else:
        return dtype

def format_array_declaration(dtype, uid, shape, alignment=None, section=None):
    attributes = _variable_attributes(alignment, section)
    s = f'{_alias_dtype(dtype)} {uid}'
    for dim in shape:
        s += f'[{dim}]'
    if attributes:
        s += f' {attributes};'
    else:
        s += ';'
    return s

def flatten(array):
    """Flatten various array types with a homogeneous API.

    Args:
        array: Can be a Numpy array, a PyTorch tensor or a nested list.
    """
    if isinstance(array, np.ndarray):
        return array.flatten()
    elif isinstance(array, torch.Tensor):
        return array.numpy().flatten()
    elif isinstance(array, list):
        return np.array(array).flatten()
    # if scalar return it as a list
    elif isinstance(array, np.generic):
        return np.array([array]).flatten()
    else:
        raise TypeError(f"Unsupported type: {type(array)}")

def format_array_initializer(dtype, array):
    s = '{\n'
    array = flatten(array)
    for el in array:
        if dtype == '__fp8':
            el_str = f'{hex(el.bits())}'
        else:
            el_str = f'{el}'
        s += f'\t{el_str},\n'
    s += '}'
    return s

# In the case of dtype __fp8, array field expects a dictionary of
# sign, exponent and mantissa arrays
def format_array_definition(dtype, uid, array, alignment=None, section=None):
    # Definition starts with the declaration stripped off of the terminating semicolon
    s = format_array_declaration(dtype, uid, array.shape, alignment, section)[:-1]
    s += ' = '
    s += format_array_initializer(dtype, array)
    s += ';'
    return s

def format_struct_definition(dtype, uid, map):
    def format_value(value):
        if isinstance(value, list):
            return format_array_initializer('str', value)
        elif isinstance(value, bool):
            return str(int(value))
        else:
            return str(value)

    filtered_map = {key: value for key, value in map.items() if value is not None and value != ''}

    formatted_items = [
        f'\t.{key} = {format_value(value)}'
        for key, value in filtered_map.items()
    ]
    s = f'{_alias_dtype(dtype)} {uid} = {{\n'
    s += ',\n'.join(formatted_items)
    s += '\n};'
    return s

def _integer_precision_t(prec):
    if isinstance(prec, str):
        return {'FP64': 8, 'FP32': 4, 'FP16': 2, 'FP8': 1}[prec]
    else:
        return prec


def size_from_precision_t(prec):
    """Return the size in bytes of a `precision_t` type.

    Args:
        prec: A value of type `precision_t`. Accepts both enum strings
            (e.g. "FP64") and integer enumeration values (e.g. 8).
    """
    return _integer_precision_t(prec)

def ctype_from_precision_t(prec):
    """Convert `precision_t` type to a C type string.

    Args:
        prec: A value of type `precision_t`. Accepts both enum strings
            (e.g. "FP64") and integer enumeration values (e.g. 8).
    """
    precision_t_to_ctype_map = {
        8: 'double',
        4: 'float',
        2: '__fp16',
        1: '__fp8'
    }
    return precision_t_to_ctype_map[_integer_precision_t(prec)]

class DataGen:
    """Base data generator class.

    Base data generator class which can be inherited to easily develop a
    custom data generator script for any kernel.
    """

    def parser(self):
        """Default argument parser for data generation scripts.

        It is an instance of the `ArgumentParser` class from the `argparse`
        module. Subclasses can extend this and add custom arguments via
        the parser's `add_argument()` method.
        """
        parser = argparse.ArgumentParser(description='Generate data for kernels')
        parser.add_argument(
            "-c", "--cfg",
            type=pathlib.Path,
            required=True,
            help='Select param config file kernel'
        )
        parser.add_argument(
            '--section',
            type=str,
            help='Section to store matrices in')
        parser.add_argument(
            'output',
            type=pathlib.Path,
            help='Path of the output header file')
        return parser

    def parse_args(self):
        """Parse default data generation script arguments.

        Returns the arguments passed to the data generation script, parsed
        using the `parser()` method.
        """
        return self.parser().parse_args()

    def emit_header(self, **kwargs):
        """Emits a C header containing generated data.

        The base implementation emits a string which only contains a
        license header. Subclasses should extend this method and append
        the generated data to the license header.

        Returns:
            A string with the generated C header contents.
        """
        return emit_license()

    def main(self):
        """Default main function for data generation scripts."""
        args = self.parse_args()

        # Load param config file
        with args.cfg.open() as f:
            param = json5.loads(f.read())
        param['section'] = args.section

        # Emit header file
        with open(args.output, 'w') as f:
            f.write(self.emit_header(**param))

def validate_tcdm_footprint(size, silent=False):
    """Check whether data of specified size fits in TCDM.

    Throws an assertion error if the specified size exceeds the space
    available for the heap in TCDM.

    Args:
        size: The size of the data in bytes.
        silent: If True, will not print the size to stdout.
    """
    assert size < TCDM_HEAP_SIZE, \
        f'Total heap space required {humanize.naturalsize(size, binary=True)} exceeds ' \
        f'limit of {humanize.naturalsize(TCDM_HEAP_SIZE, binary=True)}'
    if not silent:
        print(f'Total heap space required {humanize.naturalsize(size, binary=True)}')


def numpy_type_from_precision_t(prec):
    """Convert `precision_t` type to PyTorch type.
    Args:
        prec: A value of type `precision_t`. Accepts both enum strings
            (e.g. "FP64") and integer enumeration values (e.g. 8).
    """
    # Types which have a direct correspondence in Numpy
    precision_t_to_numpy_type_map = {
        8: np.float64,
        4: np.float32,
        2: np.float16
    }
    prec = _integer_precision_t(prec)
    assert prec != 1, "No direct correspondence between FP8 and Numpy"
    return precision_t_to_numpy_type_map[prec]

def ff_desc_from_precision_t(prec):
    """Convert `precision_t` type to a FlexFloat descriptor.

    Args:
        prec: A value of type `precision_t`. Accepts both enum strings
            (e.g. "FP64") and integer enumeration values (e.g. 8).
    """
    precision_t_to_ff_desc_map = {
        8: 'fp64',
        4: 'fp32',
        2: 'fp16',
        1: 'e5m2'
    }
    return precision_t_to_ff_desc_map[_integer_precision_t(prec)]


def generate_random_array(size, prec='FP64'):
    """Consistent random array generation for Snitch experiments.
    Samples values between -1 and 1 from a uniform distribution and
    of the exact specified type, e.g. actual 64-bit doubles.
    This function ensures that e.g. power measurements are not skewed
    by using integer values in the FPU.
    Args:
        size: Tuple of array dimensions.
        prec: A value of type `precision_t`. Accepts both enum strings
            (e.g. "FP64") and integer enumeration values (e.g. 8).
    """
    # Generate in 64b precision and then cast down
    rand = np.random.default_rng().random(size=size, dtype=np.float64) * 2 - 1
    # Generate FlexFloat array for 8b floats, casted from 16b Numpy array
    if _integer_precision_t(prec) == 1:
        return ff.array(rand.astype(np.float16), ff_desc_from_precision_t(prec))
    else:
        return rand.astype(numpy_type_from_precision_t(prec))