#!/usr/bin/env python3
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import re
import sys

from snitch.util.sim.data_utils import DataGen, format_struct_definition
from snitch.dnn import layernorm
from snitch.blas import gemm


class MLPDataGen(DataGen):

    def emit_header(self, **kwargs):
        header = [super().emit_header()]

        # LayerNorm data generation
        ln_gen = layernorm.LayernormDataGen()
        ln_gen.load_params(kwargs['layernorm'])
        ln_gen.validate()

        layernorm_i = ln_gen.generate_ifmap()
        layernorm_o = ln_gen.golden_model(layernorm_i)
        header += [ln_gen.emit_ifmap('layernorm_i', layernorm_i)]
        header += [ln_gen.emit_ofmap('layernorm_o', layernorm_o)]
        header += [ln_gen.emit_layer_struct('layernorm_cfg')]

        # FusedLinearGelu data generation
        gemm_gen = gemm.GemmDataGen()
        gemm_gen.load_params(kwargs['fused_linear_gelu'])
        gemm_gen.validate()

        _, flg_weights, flg_o = gemm_gen.generate_inputs()
        gemm_gen.a_uid = 'layernorm_o'
        header += [gemm_gen.emit_b('flg_weights', flg_weights)]
        header += [gemm_gen.emit_c('flg_o', flg_o, decl_only=True)]
        flg_struct = gemm_gen.emit_layer_struct('flg_cfg')
        # Temporarily replace gemm_fp with flg_fp, until this is actually properly solved...
        # TODO(vivianep)
        flg_struct = re.sub(r'(\.gemm_fp\s*=\s*)gemm_(\w+)', r'\1fused_linear_gelu_\2', flg_struct)
        header += [flg_struct]

        # GEMM data generation
        gemm_gen = gemm.GemmDataGen()
        gemm_gen.load_params(kwargs['gemm'])
        gemm_gen.validate()

        _, linear_weights, linear_o = gemm_gen.generate_inputs()
        gemm_gen.a_uid = 'flg_o'
        header += [gemm_gen.emit_b('linear_weights', linear_weights)]
        header += [gemm_gen.emit_c('linear_o', linear_o, decl_only=True)]
        header += [gemm_gen.emit_layer_struct('linear_cfg')]

        # MLP Configuration
        mlp_args = {
            'layernorm_cfg': '&layernorm_cfg',
            'flg_cfg': '&flg_cfg',
            'linear_cfg': '&linear_cfg',
        }
        header += [format_struct_definition('mlp_args_t', 'mlp_args', mlp_args)]
        header = '\n\n'.join(header)
        return header


if __name__ == '__main__':
    sys.exit(MLPDataGen().main())
