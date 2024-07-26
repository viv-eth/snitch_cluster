// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

#include "blas.h"
#include "snrt.h"

typedef struct {
    layernorm_layer_t *layernorm_cfg;
    gemm_args_t *flg_cfg;
    gemm_args_t *linear_cfg;
} mlp_args_t;

static inline void mlp_layer(mlp_args_t *mlp_args) {
    
    // LayerNorm of MLP
    layernorm_layer(*mlp_args->layernorm_cfg);
    snrt_cluster_hw_barrier();
    DUMP(0x88888888);

    // Fused Linear GELU
    fused_linear_gelu(mlp_args->flg_cfg);
    snrt_cluster_hw_barrier();
    DUMP(0x99999999);

    // GEMM
    gemm(mlp_args->linear_cfg);
    snrt_cluster_hw_barrier();
    DUMP(0xAAAAAAAA);
}
