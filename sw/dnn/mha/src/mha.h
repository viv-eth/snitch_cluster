// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Viviane Potocnik <vivianep@iis.ee.ethz.ch>

#include "blas.h"
#include "snrt.h"

/*
 * @struct mha_args_t
 * @brief This structure contains all parameters necessary
 *        for computing the Multi-Head Attention layer
 *
 * @var mha_args_t::layernorm_layer_cfg
 * Parameters for the LayerNorm layer
 * @var mha_args_t::gemm_layer_cfg
 * Parameters for the GEMM layer
 * @var mha_args_t::x1_input
 * Pointer to the input feature map X1
 * @var mha_args_t::x1_output
 * Pointer to the output feature map X1 after LayerNorm
 * @var mha_args_t::x2_input
 * Pointer to the input feature map X2
 * @var mha_args_t::x2_output
 * Pointer to the output feature map X2 after LayerNorm
 * @var mha_args_t::W_q
 * Pointer to the Query weight matrix W_q
 * @var mha_args_t::W_k
 * Pointer to the Key weight matrix W_k
 * @var mha_args_t::W_v
 * Pointer to the Value weight matrix W_v
 */

typedef struct {
    struct layernorm_layer_struct *layernorm_cfg;
    struct gemm_args_struct *gemm_cfg;
    struct flashattention_2_layer_struct *flashattention_2_cfg;
    struct fused_concat_linear_layer_struct *fcl_cfg;
    void *x1_input;
    void *x1_output;
    void *x2_input;
    void *x2_output;
    void *W_q;
    void *Q;
    void *W_k;
    void *K;
    void *W_v;
    void *V;
    void *O_fa;
    void *O;
} mha_args_t;

static inline int mha_layer(mha_args_t *mha_args) {
    // LayerNorm of X1 input
    layernorm_layer_t layernorm_layer_cfg = {
        .batch_size = mha_args->layernorm_cfg->batch_size,
        .seq_len = mha_args->layernorm_cfg->seq_len,
        .embeddings = mha_args->layernorm_cfg->embeddings,
        .n_tiles = mha_args->layernorm_cfg->n_tiles,
        .implementation = mha_args->layernorm_cfg->implementation,
        .eps = mha_args->layernorm_cfg->eps,
        .ifmap = mha_args->x1_input,
        .ofmap = mha_args->x1_output,
        .dtype = mha_args->layernorm_cfg->dtype};

    layernorm_layer(layernorm_layer_cfg);
    snrt_cluster_hw_barrier();

    // LayerNorm of X2 input
    layernorm_layer_cfg.ifmap = mha_args->x2_input;
    layernorm_layer_cfg.ofmap = mha_args->x2_output;

    layernorm_layer(layernorm_layer_cfg);
    snrt_cluster_hw_barrier();

    // GEMM: X1_output * W_q
    mha_args->gemm_cfg->a = mha_args->x1_output;
    mha_args->gemm_cfg->b = mha_args->W_q;

    gemm(mha_args->gemm_cfg);
    snrt_cluster_hw_barrier();

    mha_args->Q = mha_args->gemm_cfg->c;

    // GEMM: X2_output * W_k
    mha_args->gemm_cfg->a = mha_args->x2_output;
    mha_args->gemm_cfg->b = mha_args->W_k;

    gemm(mha_args->gemm_cfg);
    snrt_cluster_hw_barrier();

    mha_args->K = mha_args->gemm_cfg->c;

    // GEMM: X2_output * W_v
    mha_args->gemm_cfg->a = mha_args->x2_output;
    mha_args->gemm_cfg->b = mha_args->W_v;

    gemm(mha_args->gemm_cfg);
    snrt_cluster_hw_barrier();

    mha_args->V = mha_args->gemm_cfg->c;

    snrt_cluster_hw_barrier();

    // FlashAttention-2: Q, K, V
    mha_args->flashattention_2_cfg->Q = mha_args->Q;
    mha_args->flashattention_2_cfg->K = mha_args->K;
    mha_args->flashattention_2_cfg->V = mha_args->V;
    mha_args->flashattention_2_cfg->gemm_implementation =
        mha_args->gemm_cfg->gemm_fp;

    flashattention_2_layer(*(mha_args->flashattention_2_cfg));
    snrt_cluster_hw_barrier();
    mha_args->O_fa = mha_args->flashattention_2_cfg->O;
    snrt_cluster_hw_barrier();

    // Fused Concat Linear Layer
    mha_args->fcl_cfg->inputs[snrt_cluster_idx()] = mha_args->O_fa;
    fused_concat_linear_layer(*(mha_args->fcl_cfg));
    snrt_cluster_hw_barrier();

    // MHA output
    mha_args->O = mha_args->fcl_cfg->linear_output;

    return 0;
}