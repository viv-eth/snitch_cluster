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
import numpy as np

import pyflexfloat as ff

from snitch.util.sim import data_utils
from snitch.util.sim.data_utils import DataGen, format_array_declaration, \
    format_struct_definition, format_array_definition

torch.manual_seed(42)


def golden_model(ifmap, eps):
    # Compute the mean and variance considering the last dimension (embeddings)
    # The dimensions for mean and variance calculations are (-1,), meaning the last dimension.
    # Keep the dimensions for broadcasting in normalization.
    mean = np.mean(ifmap, axis=(-1,), keepdims=True)
    diff = ifmap - mean
    var = np.mean(diff*diff, axis=(-1,), keepdims=True)

    # Normalize the input tensor
    ofmap = (ifmap - mean) / np.sqrt(var + eps)

    return ofmap


class LayernormDataGen(DataGen):

    # AXI splits bursts crossing 4KB address boundaries. To minimize
    # the occurrence of these splits the data should be aligned to 4KB
    BURST_ALIGNMENT = 4096

    def load_params(self, params):
        self.batch_size = params['input_dim']['batch_size']
        self.seq_len = params['input_dim']['seq_len']
        self.embeddings = params['input_dim']['embeddings']
        self.prec = params['prec']
        self.ctype = data_utils.ctype_from_precision_t(self.prec)
        self.eps = params['eps']
        self.prec = params['prec']
        self.n_tiles = params['n_tiles']
        self.implementation = params['implementation']

    def golden_model(self, ifmap):
        return golden_model(ifmap, self.eps)

    def validate(self):
        # Calculate total TCDM occupation
        prec = data_utils.size_from_precision_t(self.prec)
        tiled_seq_len = self.seq_len / self.n_tiles
        total_size = self.batch_size * tiled_seq_len * self.embeddings * prec
        data_utils.validate_tcdm_footprint(total_size)

        assert self.seq_len % self.n_tiles == 0, 'Input dimension is not' \
                                                 ' an integer multiple of' \
                                                 ' tile size'
        assert self.prec != "FP64", 'FP64 not supported'
        assert not (self.implementation == "BASELINE"), 'No baseline implementations' \
                                                        ' (switch to NAIVE)'
        assert not (self.implementation == "OPT_EX"), 'Expanding layernorm kernels not supported'
        assert not (self.prec == "FP8" and self.implementation == "NAIVE"), 'FP8 not ' \
                                                                            'supported in' \
                                                                            'naive ' \
                                                                            'implementation'

    def generate_ifmap(self):
        ff_desc = data_utils.ff_desc_from_precision_t(self.prec)
        return ff.array(np.random.rand(self.batch_size, self.seq_len, self.embeddings), ff_desc)

    def emit_ifmap(self, uid, data):
        self.ifmap_uid = uid
        return format_array_definition(self.ctype, uid, data,
                                       alignment=self.BURST_ALIGNMENT)

    def emit_ofmap(self, uid, data):
        self.ofmap_uid = uid
        return format_array_declaration(self.ctype, uid, data.shape,
                                        alignment=self.BURST_ALIGNMENT)

    def emit_layer_struct(self, uid):
        layer_cfg = {
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'embeddings': self.embeddings,
            'n_tiles': self.n_tiles,
            'implementation': self.implementation,
            'ifmap': self.ifmap_uid,
            'ofmap': self.ofmap_uid,
            'eps': self.eps,
            'dtype': self.prec
        }
        return format_struct_definition('layernorm_layer_t', uid, layer_cfg)

    def emit_header(self, **kwargs):
        header = [super().emit_header()]

        self.load_params(kwargs)
        self.validate()

        ifmap = self.generate_ifmap()
        ofmap = self.golden_model(ifmap)

        header += [self.emit_ifmap('ifmap', ifmap)]
        header += [self.emit_ofmap('ofmap', ofmap)]
        header += [self.emit_layer_struct('layer')]
        header = '\n\n'.join(header)

        return header


if __name__ == '__main__':
    sys.exit(LayernormDataGen().main())
