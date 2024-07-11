// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Viviane Potocnik <vivianep@iis.ee.ethz.ch>

#include "blas.h"
#include "snrt.h"

/*
 * @struct mlp_args_t
 * @brief This structure contains all parameters necessary
 *        for computing the Multi-layer Perceptron (MLP) layer.
 * @var mlp_args_t::layernorm_layer_cfg
 * Parameters for the LayerNorm layer
 * @var mlp_args_t::flg_cfg
 * Parameters for the Fused Linear GELU layer
 * @var mlp_args_t::gemm_cfg
 * Parameters for the GEMM layer
 */

typedef struct {
    struct layernorm_layer_struct *layernorm_cfg;
    struct flg_args_struct *flg_cfg;
    struct gemm_args_struct *gemm_cfg;
    void* mlp_input;
    void* flg_output;
    void* mlp_output;
    void* W_ff;
    void* W_mlp;
} mlp_args_t;

static inline int mlp_layer(mlp_args_t *mlp_args) {
    
    // LayerNorm of MLP
    layernorm_layer_t layernorm_layer_cfg = {
        .batch_size = mlp_args->layernorm_cfg->batch_size,
        .seq_len = mlp_args->layernorm_cfg->seq_len,
        .embeddings = mlp_args->layernorm_cfg->embeddings,
        .n_tiles = mlp_args->layernorm_cfg->n_tiles,
        .implementation = mlp_args->layernorm_cfg->implementation,
        .eps = mlp_args->layernorm_cfg->eps,
        .ifmap = mlp_args->mlp_input,
        .ofmap = mlp_args->mlp_input,
        .dtype = mlp_args->layernorm_cfg->dtype,
    };
    layernorm_layer_cfg.ifmap = mlp_args->mlp_input;
    layernorm_layer_cfg.ofmap = mlp_args->mlp_input;
    layernorm_layer(layernorm_layer_cfg);
    snrt_cluster_hw_barrier();

    // Fused Linear GELU
    mlp_args->flg_cfg->a = mlp_args->mlp_input;
    mlp_args->flg_cfg->b = mlp_args->W_ff;
    fused_linear_gelu(mlp_args->flg_cfg);
    snrt_cluster_hw_barrier();

    mlp_args->flg_output = mlp_args->flg_cfg->c;
    snrt_cluster_hw_barrier();

    // GEMM
    mlp_args->gemm_cfg->a = mlp_args->flg_output;
    mlp_args->gemm_cfg->b = mlp_args->W_mlp;
    gemm(mlp_args->gemm_cfg);
    snrt_cluster_hw_barrier();

    mlp_args->mlp_output = mlp_args->gemm_cfg->c;
    snrt_cluster_hw_barrier();

    return 0;
}