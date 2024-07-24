#!/usr/bin/env python3
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import numpy as np
import torch
import sys

from snitch.util.sim.data_utils import DataGen, format_struct_definition
from snitch.dnn import layernorm, flashattention_2, fused_concat_linear
from snitch.blas import gemm


class MHADataGen(DataGen):

    def emit_header(self, **kwargs):
        header = [super().emit_header()]

        # LayerNorm data generation
        ln_gen = layernorm.LayernormDataGen()
        ln_gen.load_params(kwargs['layernorm'])
        ln_gen.validate()

        X1_i = ln_gen.generate_ifmap()
        X1_o = ln_gen.golden_model(X1_i)
        header += [ln_gen.emit_ifmap('X1_i', X1_i)]
        header += [ln_gen.emit_ofmap('X1_o', X1_o)]
        header += [ln_gen.emit_layer_struct('layernorm1_cfg')]

        X2_i = ln_gen.generate_ifmap()
        X2_o = ln_gen.golden_model(X2_i)
        header += [ln_gen.emit_ifmap('X2_i', X2_i)]
        header += [ln_gen.emit_ofmap('X2_o', X2_o)]
        header += [ln_gen.emit_layer_struct('layernorm2_cfg')]

        # GEMM data generation
        gemm_gen = gemm.GemmDataGen()
        gemm_gen.load_params(kwargs['gemm'])
        gemm_gen.validate()

        _, W_q, Q = gemm_gen.generate_inputs()
        gemm_gen.a_uid = 'X1_i'
        header += [gemm_gen.emit_b('W_q', W_q)]
        header += [gemm_gen.emit_c('Q', Q, decl_only=True)]
        header += [gemm_gen.emit_layer_struct('gemmQ_cfg')]

        _, W_k, K = gemm_gen.generate_inputs()
        gemm_gen.a_uid = 'X2_i'
        header += [gemm_gen.emit_b('W_k', W_k)]
        header += [gemm_gen.emit_c('K', K, decl_only=True)]
        header += [gemm_gen.emit_layer_struct('gemmK_cfg')]

        _, W_v, V = gemm_gen.generate_inputs()
        gemm_gen.a_uid = 'X2_i'
        header += [gemm_gen.emit_b('W_v', W_v)]
        header += [gemm_gen.emit_c('V', V, decl_only=True)]
        header += [gemm_gen.emit_layer_struct('gemmV_cfg')]

        # FlashAttention-2 data generation
        fa2_gen = flashattention_2.FlashAttention2DataGen()
        fa2_gen.load_params(kwargs['flashattention_2'])
        fa2_gen.validate()

        fa2_o = fa2_gen.golden_model(Q, K, V)
        fa2_gen.q_uid = 'Q'
        fa2_gen.k_uid = 'K'
        fa2_gen.v_uid = 'V'
        if kwargs['flashattention_2']['use_mask']:
            mask = fa2_gen.generate_mask()
            header += [fa2_gen.emit_mask('mask', mask)]
        header += [fa2_gen.emit_ofmap('fa2_o', fa2_o)]
        header += [fa2_gen.emit_layer_struct('flashattention_2_cfg')]

        # FusedConcatLinear data generation
        fcl_gen = fused_concat_linear.FusedConcatLinearDataGen()
        fcl_gen.load_params(kwargs['fused_concat_linear'])
        fcl_gen.validate()
        
        _, weights = fcl_gen.generate_inputs()
        inputs = [torch.from_numpy(fa2_o.astype(np.float32))]
        concat_o, linear_o = fcl_gen.golden_model(inputs, weights)
        fcl_gen.inputs_uid = 'fa2_o'
        header += [fcl_gen.emit_weights('fcl_weights', weights)]
        header += [fcl_gen.emit_concat_o('concat_output', concat_o)]
        header += [fcl_gen.emit_linear_o('linear_output', linear_o)]
        header += [fcl_gen.emit_layer_struct('fcl_cfg')]

        # MHA Configuration
        mha_args_cfg = {
            'layernorm1_cfg': '&layernorm1_cfg',
            'layernorm2_cfg': '&layernorm2_cfg',
            'gemmQ_cfg': '&gemmQ_cfg',
            'gemmK_cfg': '&gemmK_cfg',
            'gemmV_cfg': '&gemmV_cfg',
            'flashattention_2_cfg': '&flashattention_2_cfg',
            'fcl_cfg': '&fcl_cfg'
        }
        header += [format_struct_definition('mha_args_t', 'mha_args', mha_args_cfg)]
        header = '\n\n'.join(header)

        return header


if __name__ == '__main__':
    sys.exit(MHADataGen().main())
