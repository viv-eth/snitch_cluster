#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Tim Fischer <fischeti@iis.ee.ethz.ch>
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import sys
import torch

from snitch.util.sim import data_utils
from snitch.util.sim.data_utils import DataGen, format_struct_definition, \
    format_array_definition, format_array_declaration

torch.manual_seed(42)


# Sigmoid based approximation of the GeLU activation function
# adapted from i-BERT (https://arxiv.org/pdf/2101.01321.pdf)
# L(x) = sgn(x) [a(clip(|x|, max = −b) + b)^2 + 1]
# a = -0.2888, b = -1.769
def sigmoid_gelu(x):
    a = -0.2888
    b = -1.769
    return torch.sign(x) * (a * (torch.clamp(torch.abs(x), max=-b) + b)**2 + 1)


class GeluDataGen(DataGen):

    def load_params(self, params):
        self.size = params['size']
        self.dtype = params['dtype']
        self.torch_type = data_utils.torch_type_from_precision_t(self.dtype)
        self.ctype = data_utils.ctype_from_precision_t(self.dtype)

    def golden_model(self, ifmap):
        gelu = torch.nn.GELU(approximate='tanh')
        # gelu = sigmoid_gelu
        return gelu(ifmap)

    def generate_ifmap(self):
        return torch.randn(self.size, requires_grad=False, dtype=self.torch_type)

    def emit_ifmap(self, uid, data):
        self.ifmap_uid = uid
        return format_array_definition(self.ctype, uid, data)

    def emit_ofmap(self, uid, data):
        self.ofmap_uid = uid
        return format_array_declaration(self.ctype, uid, data.shape)

    def emit_layer_struct(self, uid):
        layer_cfg = {
            'size':  self.size,
            'ifmap': self.ifmap_uid,
            'ofmap': self.ofmap_uid,
            'dtype': self.dtype
        }
        return format_struct_definition('gelu_layer_t', uid, layer_cfg)

    def emit_header(self, **kwargs):
        header = [super().emit_header()]

        self.load_params(kwargs)

        ifmap = self.generate_ifmap()
        ofmap = self.golden_model(ifmap)

        header += [self.emit_ifmap('ifmap', ifmap)]
        header += [self.emit_ofmap('ofmap', ofmap)]
        header += [self.emit_layer_struct('layer')]
        header = '\n\n'.join(header)

        return header


if __name__ == '__main__':
    sys.exit(GeluDataGen().main())
