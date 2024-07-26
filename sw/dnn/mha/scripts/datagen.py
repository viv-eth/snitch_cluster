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

from snitch.util.sim.data_utils import DataGen, format_struct_definition,\
                                       format_cfgs_array

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

        # Start of MHA data generation
        num_heads = kwargs['num_heads']
        # GEMM data generation
        gemm_gen = gemm.GemmDataGen()
        gemm_gen.load_params(kwargs['gemm'])
        gemm_gen.validate()

        gemm_gen.a_uid = 'X1_i'
        # array of num_heads elements, each element is a struct
        gemmQ_cfgs = np.array([])
        Q_matrices = []
        for head in range(num_heads):
            _, W_q, Q = gemm_gen.generate_inputs()
            Q_matrices.append(Q)
            header += [gemm_gen.emit_b(f'W_q_{head}', W_q)]
            header += [gemm_gen.emit_c(f'Q_{head}', Q, decl_only=True)]
            gemmQ_cfgs = np.append(gemmQ_cfgs, gemm_gen.emit_layer_struct('gemmQ_cfg', return_cfg=True))
        header += [format_cfgs_array('gemm_args_t', 'gemmQ_cfgs', gemmQ_cfgs, num_heads)]

        gemm_gen.a_uid = 'X2_i'
        gemmK_cfgs = np.array([])
        K_matrices = []
        for head in range(num_heads):
            _, W_k, K = gemm_gen.generate_inputs()
            K_matrices.append(K)
            header += [gemm_gen.emit_b(f'W_k_{head}', W_k)]
            header += [gemm_gen.emit_c(f'K_{head}', K, decl_only=True)]
            gemmK_cfgs = np.append(gemmK_cfgs, gemm_gen.emit_layer_struct('gemmK_cfg', return_cfg=True))
        header += [format_cfgs_array('gemm_args_t', 'gemmK_cfgs', gemmK_cfgs, num_heads)]

        gemm_gen.a_uid = 'X2_i'
        gemmV_cfgs = np.array([])
        V_matrices = []
        for head in range(num_heads):
            _, W_v, V = gemm_gen.generate_inputs()
            V_matrices.append(V)
            header += [gemm_gen.emit_b(f'W_v_{head}', W_v)]
            header += [gemm_gen.emit_c(f'V_{head}', V, decl_only=True)]
            gemmV_cfgs = np.append(gemmV_cfgs, gemm_gen.emit_layer_struct('gemmV_cfg', return_cfg=True))
        header += [format_cfgs_array('gemm_args_t', 'gemmV_cfgs', gemmV_cfgs, num_heads)]

        # FlashAttention-2 data generation
        fa2_gen = flashattention_2.FlashAttention2DataGen()
        fa2_gen.load_params(kwargs['flashattention_2'])
        fa2_gen.validate()

        fa2_cfgs = np.array([])
        fa2_o_matrices = []
        for head in range(num_heads):
            Q = Q_matrices[head]
            K = K_matrices[head]
            V = V_matrices[head]
            fa2_o = fa2_gen.golden_model(Q, K, V)
            fa2_gen.q_uid = f'Q_{head}'
            fa2_gen.k_uid = f'K_{head}'
            fa2_gen.v_uid = f'V_{head}'
            if kwargs['flashattention_2']['use_mask']:
                mask = fa2_gen.generate_mask()
                header += [fa2_gen.emit_mask(f'mask_{head}', mask)]
            header += [fa2_gen.emit_ofmap(f'fa2_o_{head}', fa2_o)]
            fa2_o_matrices.append(fa2_o)
            fa2_cfgs = np.append(fa2_cfgs, fa2_gen.emit_layer_struct('flashattention_2_cfg', return_cfg=True))
        header += [format_cfgs_array('flashattention_2_layer_t', 'fa2_cfgs', fa2_cfgs, num_heads)]

        # FusedConcatLinear data generation
        fcl_gen = fused_concat_linear.FusedConcatLinearDataGen()
        fcl_gen.load_params(kwargs['fused_concat_linear'])
        fcl_gen.validate()
        
        fcl_weights = []
        inputs = []
        inputs_uids = []
        weight_uids = []
        for head in range(num_heads):
            _, weights = fcl_gen.generate_inputs()
            inputs.append(torch.from_numpy(fa2_o_matrices[head].astype(np.float32)))
            fcl_weights.append(weights)
            header += [fcl_gen.emit_weights(f'fcl_weights_{head}', weights)]
            # fcl_gen.inputs_uid = f'fa2_o_{head}'
            inputs_uids.append(f'fa2_o_{head}')
            weight_uids.append(f'fcl_weights_{head}')
        fcl_gen.inputs_uid = inputs_uids
        fcl_gen.weights_uid = weight_uids
        fcl_gen.num_inputs = num_heads
        
        # inputs = [torch.from_numpy(fa2_o_matrices[head].astype(np.float32))]
        concat_o, linear_o = fcl_gen.golden_model(inputs, weights)
        header += [fcl_gen.emit_inputs('inputs', np.array(inputs))]
        header += [fcl_gen.emit_weights('weights', np.array(fcl_weights))]
        header += [fcl_gen.emit_concat_o(f'concat_output', concat_o)]
        header += [fcl_gen.emit_linear_o(f'linear_output', linear_o)]
        header += [fcl_gen.emit_concat_o('concat_output', concat_o)]
        header += [fcl_gen.emit_linear_o('linear_output', linear_o)]
        header += [fcl_gen.emit_layer_struct('fcl_cfg')]

        # MHA Configuration
        mha_args_cfg = {
            'num_heads': num_heads,
            'layernorm1_cfg': '&layernorm1_cfg',
            'layernorm2_cfg': '&layernorm2_cfg',
            'gemmQ_cfgs': '&gemmQ_cfgs[{}]'.format(num_heads),
            'gemmK_cfgs': '&gemmK_cfgs[{}]'.format(num_heads),
            'gemmV_cfgs': '&gemmV_cfgs[{}]'.format(num_heads),
            'flashattention_2_cfgs': '&fa2_cfgs[{}]'.format(num_heads),
            'fcl_cfg': '&fcl_cfg'
        }
        header += [format_struct_definition('mha_args_t', 'mha_args', mha_args_cfg)]
        header = '\n\n'.join(header)

        return header


if __name__ == '__main__':
    sys.exit(MHADataGen().main())
