#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Tim Fischer     <fischeti@iis.ee.ethz.ch>
#          Luca Bertaccini <lbertaccini@iis.ee.ethz.ch>
#          Viviane Potocnik <vivianep@iis.ee.ethz.ch>
#          Luca Colagrande <colluca@iis.ee.ethz.ch>

import numpy as np
import re
import pyflexfloat as ff
import sys
from snitch.util.sim import data_utils
from snitch.util.sim.data_utils import DataGen, format_array_declaration, \
    format_struct_definition, format_array_definition


np.random.seed(42)


def golden_model(alpha, a, b, beta, c):
    return alpha * np.matmul(a, b) + beta * c


def exact_golden_model(a, b, beta, c):
    M, N, K = a.shape[0], b.shape[1], b.shape[0]
    result = beta * c
    for m in range(M):
        for n in range(N):
            for k in range(K):
                result[m][n] += a[m][k] * b[k][n]
    return result


class GemmDataGen(DataGen):

    def golden_model(self, a, b, c):
        return exact_golden_model(a, b, self.beta, c)

    def infer_implementation(self, gemm_fp):
        # gemm_fp: "gemm_fp64_opt"
        # create a regex with fp_<type>_<implementation>
        prec, impl = re.search(r'gemm_fp(\d+)_(\w+)', gemm_fp).group(1, 2)
        return (int(prec) / 8), impl

    def load_params(self, params):
        self.M = params.get('M')
        self.N = params.get('N')
        self.K = params.get('K')
        self.m_tiles = params.get('m_tiles')
        self.n_tiles = params.get('n_tiles')
        self.k_tiles = params.get('k_tiles')
        self.load_a = params.get('load_a')
        self.load_b = params.get('load_b')
        self.load_c = params.get('load_c')
        self.setup_ssr = params.get('setup_ssr')
        self.parallelize_m = params.get('parallelize_m')
        self.parallelize_k = params.get('parallelize_k')
        self.gemm_fp = params.get('gemm_fp')
        self.transa = params.get('transa')
        self.transb = params.get('transb')
        self.alpha = params.get('alpha', 1)
        self.beta = params.get('beta')
        self.section = params.get('section')
        self.dtype, self.impl = self.infer_implementation(self.gemm_fp)
        self.prec = data_utils.size_from_precision_t(self.dtype)
        self.ff_desc = data_utils.ff_desc_from_precision_t(self.dtype)
        self.ctype = data_utils.ctype_from_precision_t(self.dtype)

    def validate(self):
        frac_m = self.M / self.m_tiles
        frac_n = self.N / self.n_tiles
        frac_k = self.K / self.k_tiles

        # Calculate total TCDM occupation
        # Note: doesn't account for double buffering
        a_size = frac_m * frac_k * self.prec
        b_size = frac_k * frac_n * self.prec
        c_size = frac_m * frac_n * self.prec
        total_size = a_size
        total_size += b_size
        total_size += c_size
        data_utils.validate_tcdm_footprint(total_size)

        assert self.alpha == 1, 'alpha != 1 not supported'
        assert (self.M % self.m_tiles) == 0, 'M is not an integer multiple of tile size'
        assert (self.N % self.n_tiles) == 0, 'N is not an integer multiple of tile size'
        assert (self.K % self.k_tiles) == 0, 'K is not an integer multiple of tile size'
        assert not (self.parallelize_m and self.parallelize_k), 'Cannot parallelize K and M' \
            'simultaneously'
        assert not self.transa, 'SIMD kernels don\'t support transposed A matrix'
        assert (self.prec == 8) or (self.impl == 'baseline') or (self.impl == 'naive') \
            or self.transb, 'Optimized SIMD kernels only support transposed B matrix'
        assert not self.transb or self.n_tiles == 1, 'Tiling in the N dimension not supported' \
            ' if B is transposed'
        assert not self.transb or self.k_tiles == 1, 'Tiling in the K dimension not supported' \
            ' if B is transposed'
        assert (self.impl == 'baseline') or (self.impl == 'naive') or frac_n >= 8, \
            'N dimension of tile size must be greater or equal to the unrolling factor (8) ' \
            'when using optimized kernels'
        assert self.beta == 0 or self.beta == 1, 'Only values of 0 or 1 supported for beta'
        assert not (self.prec == 8 and self.impl == "baseline"), 'No baseline implemented' \
            ' for FP64 (switch to NAIVE)'
        assert not (((self.prec == 8) or (self.prec == 4)) and self.impl == "opt_ex"), \
            'Expanding GEMM kernels' \
            ' not supported for FP64 and FP32'
        assert not (self.prec == 1 and self.impl == "opt"), 'FP8 not supported in' \
            ' optimized implementation' \
            ' (switch to opt_ex)'

    def generate_inputs(self):
        a = ff.array(np.random.rand(self.M, self.K), self.ff_desc)
        b = ff.array(np.random.rand(self.K, self.N), self.ff_desc)
        c = ff.array(np.random.rand(self.M, self.N), self.ff_desc)
        return a, b, c

    def emit_a(self, uid, data):
        self.a_uid = uid
        # Store in transposed form if requested
        data = data.T if self.transa else data
        data = data.flatten()
        return format_array_definition(self.ctype, uid, data, section=self.section)

    def emit_b(self, uid, data):
        self.b_uid = uid
        # Store in transposed form if requested
        data = data.T if self.transb else data
        data = data.flatten()
        return format_array_definition(self.ctype, uid, data, section=self.section)

    def emit_c(self, uid, data, decl_only=False):
        self.c_uid = uid
        data = data.flatten()
        if decl_only:
            return format_array_declaration(self.ctype, uid, data.shape, section=self.section)
        else:
            return format_array_definition(self.ctype, uid, data, section=self.section)

    def emit_layer_struct(self, uid, return_cfg=False):
        layer_cfg = {
            'prec': self.prec,
            'setup_ssr': self.setup_ssr,
            'parallelize_m': self.parallelize_m,
            'parallelize_k': self.parallelize_k,
            'm_tiles': self.m_tiles,
            'n_tiles': self.n_tiles,
            'k_tiles': self.k_tiles,
            'load_a': self.load_a,
            'load_b': self.load_b,
            'load_c': self.load_c,
            'transa': self.transa,
            'transb': self.transb,
            'M': self.M,
            'N': self.N,
            'K': self.K,
            'alpha': self.alpha,
            'beta': self.beta,
            'gemm_fp': self.gemm_fp,
            'a': self.a_uid,
            'b': self.b_uid,
            'c': self.c_uid,
        }
        if return_cfg:
            return layer_cfg
        return format_struct_definition('gemm_args_t', uid, layer_cfg)

    def emit_header(self, **kwargs):
        header = [super().emit_header()]

        self.load_params(kwargs)
        self.validate()

        a, b, c = self.generate_inputs()

        header += [self.emit_a('a', a)]
        header += [self.emit_b('b', b)]
        header += [self.emit_c('c', c)]
        header += [self.emit_layer_struct('args')]
        header = '\n\n'.join(header)

        return header


if __name__ == "__main__":
    sys.exit(GemmDataGen().main())
