#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import numpy as np
import sys
import torch

from snitch.util.sim import data_utils
from snitch.util.sim.data_utils import DataGen, format_struct_definition, \
    format_array_definition, format_array_declaration, format_ifdef_wrapper

torch.manual_seed(42)


class FusedConcatLinearDataGen(DataGen):

    def load_params(self, params):
        self.num_inputs = params['num_inputs']
        self.input_shape = params['input_shape']
        self.output_shape = params['output_shape']
        self.dtype = params['dtype']
        self.gemm_implementation = params['gemm_implementation']
        self.torch_type = data_utils.torch_type_from_precision_t(self.dtype)
        self.ctype = data_utils.ctype_from_precision_t(self.dtype)

    def validate(self):
        assert self.input_shape[0] == self.output_shape[0], 'Inconsistent input and output shapes'

    def generate_inputs(self):
        inputs = [torch.rand(*self.input_shape, requires_grad=False, dtype=self.torch_type)
                  for _ in range(self.num_inputs)]
        weights = torch.rand([self.input_shape[1]*self.num_inputs, self.output_shape[1]],
                             requires_grad=False, dtype=self.torch_type)
        return inputs, weights

    def golden_model(self, inputs, weights):
        innermost_dim = len(inputs[0].shape) - 1
        concat_output = torch.cat(inputs, dim=innermost_dim)
        linear_output = torch.matmul(concat_output, weights)
        return concat_output, linear_output

    # Expects data to be an array of input tensors
    def emit_inputs(self, uid, data):
        self.inputs_uid = uid
        stmts = [format_array_definition(self.ctype, f'{uid}_{i}', t)
                 for i, t in enumerate(data)]
        stmts += [format_array_definition('void*', uid,
                  np.array([f'{uid}_{i}' for i in range(self.num_inputs)]))]
        return '\n\n'.join(stmts)

    def emit_weights(self, uid, data):
        self.weights_uid = uid
        return format_array_definition(self.ctype, uid, data)

    def emit_concat_o(self, uid, data):
        self.concat_o_uid = uid
        return format_array_declaration(self.ctype, uid, data.shape)

    def emit_linear_o(self, uid, data):
        self.linear_o_uid = uid
        return format_array_declaration(self.ctype, uid, data.shape)

    def emit_layer_struct(self, uid):
        layer_cfg = {
            'num_inputs': self.num_inputs,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'dtype': self.dtype,
            'gemm_implementation': self.gemm_implementation,
            'inputs': self.inputs_uid,
            'weights': self.weights_uid,
            'concat_output': self.concat_o_uid,
            'linear_output': self.linear_o_uid
        }
        return format_struct_definition('fused_concat_linear_layer_t', uid, layer_cfg)

    def emit_header(self, **kwargs):
        header = [super().emit_header()]

        self.load_params(kwargs)
        self.validate()

        inputs, weights = self.generate_inputs()
        concat_o, linear_o = self.golden_model(inputs, weights)

        header += [self.emit_inputs('inputs', inputs)]
        header += [self.emit_weights('weights', weights)]
        header += [self.emit_concat_o('concat_output', concat_o)]
        header += [self.emit_linear_o('linear_output', linear_o)]
        header += [self.emit_layer_struct('layer')]
        header = '\n\n'.join(header)

        return header


if __name__ == '__main__':
    sys.exit(FusedConcatLinearDataGen().main())
