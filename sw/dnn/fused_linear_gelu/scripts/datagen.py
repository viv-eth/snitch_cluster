#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Viviane Potocnik <vivianep@iis.ee.ethz.ch>
# Authors: Luca Colagrande <colluca@iis.ee.ethz.ch>

import re
import sys

from snitch.util.sim.data_utils import DataGen
from snitch.blas import gemm


class FusedLinearGeluDataGen(DataGen):

    def exact_golden_model(self, a, b, beta, c, a_gelu, b_gelu):
        M, N, K = a.shape[0], b.shape[1], b.shape[0]
        result = beta * c
        result_shape = result.shape
        for m in range(M):
            for n in range(N):
                for k in range(K):
                    result[m][n] += a[m][k] * b[k][n]
                    result[m][n] = sigmoid_gelu(result[m][n], result_shape, a_gelu, b_gelu)
        return result

    def emit_header(self, **kwargs):
        gemm_gen = gemm.GemmDataGen()
        header = gemm_gen.emit_header(**kwargs)
        # Temporarily replace gemm_fp with flg_fp, until this is actually properly solved...
        # TODO(vivianep)
        header = re.sub(r'(\.gemm_fp\s*=\s*)gemm_(\w+)', r'\1fused_linear_gelu_\2', header)
        return header


if __name__ == "__main__":
    sys.exit(FusedLinearGeluDataGen().main())
