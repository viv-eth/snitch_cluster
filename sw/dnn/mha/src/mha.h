// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

#include "blas.h"
#include "snrt.h"

typedef struct {
    layernorm_layer_t *layernorm1_cfg;
    layernorm_layer_t *layernorm2_cfg;
    gemm_args_t *gemmQ_cfg;
    gemm_args_t *gemmK_cfg;
    gemm_args_t *gemmV_cfg;
    flashattention_2_layer_t *flashattention_2_cfg;
    fused_concat_linear_layer_t *fcl_cfg;
} mha_args_t;

static inline void mha_layer(mha_args_t *mha_args) {
    // LayerNorm of X1 input
    layernorm_layer(*mha_args->layernorm1_cfg);
    snrt_cluster_hw_barrier();

    // LayerNorm of X2 input
    layernorm_layer(*mha_args->layernorm2_cfg);
    snrt_cluster_hw_barrier();

    // GEMM: X1_output * W_q
    gemm(mha_args->gemmQ_cfg);
    snrt_cluster_hw_barrier();

    // GEMM: X2_output * W_k
    gemm(mha_args->gemmK_cfg);
    snrt_cluster_hw_barrier();

    // GEMM: X2_output * W_v
    gemm(mha_args->gemmV_cfg);
    snrt_cluster_hw_barrier();

    // FlashAttention-2: Q, K, V
    flashattention_2_layer(*mha_args->flashattention_2_cfg);
    snrt_cluster_hw_barrier();

    // FusedConcatLinear Layer
    fused_concat_linear_layer(*mha_args->fcl_cfg);
    snrt_cluster_hw_barrier();
}
