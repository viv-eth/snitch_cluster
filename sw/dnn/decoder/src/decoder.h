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
} decoder_args_t;

static inline void decoder_block(decoder_args_t *decoder_args) {
    for (uint32_t i = 0; i < decoder_args->num_layers; i++) {
        mha_layer(decoder_args->mha_args);
    }
    mlp_layer(decoder_args->mlp_args);
}