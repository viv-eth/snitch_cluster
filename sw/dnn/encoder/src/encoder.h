// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Luca Colagrande <colluca@iis.ee.ethz.ch>
//         Viviane Potocnik <vivianep@iis.ee.ethz.ch>

#include "dnn.h"

typedef struct {
    uint32_t num_layers;
    mha_args_t *mha_args;
    mlp_args_t *mlp_args;
} encoder_args_t;

static inline void encoder_block(encoder_args_t *encoder_args) {
    for (uint32_t i = 0; i < encoder_args->num_layers; i++) {
        mha_layer(encoder_args->mha_args);
        mlp_layer(encoder_args->mlp_args);
    }
}
