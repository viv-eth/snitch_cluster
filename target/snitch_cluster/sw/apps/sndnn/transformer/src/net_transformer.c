// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>

// SW testbench for a transformer model
// Currently oonly supporting FP32 precision


#include "data_transformer.h"
#include "layer.h"
#include "math.h"
#include "transformer_layer.h"
#include "snrt.h"
#include "utils.h"

int main() {

    transformer_l.ifmap = (float*)transformer_ifmap_dram;
    transformer_l.Wq = (float*)transformer_weights_q_dram;
    transformer_l.Wk = (float*)transformer_weights_k_dram;
    transformer_l.Wv = (float*)transformer_weights_v_dram;

    transformer_layer(&transformer_l);

    snrt_global_barrier();

    return 0;
}