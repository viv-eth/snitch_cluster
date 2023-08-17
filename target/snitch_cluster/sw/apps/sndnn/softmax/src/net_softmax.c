// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>

// SW testbench for profiling softmax kernels in different
// floating point precisions (fp64, fp32, fp16, fp8), as well as


#include "data_softmax.h"
#include "layer.h"
#include "math.h"
#include "network.h"
#include "softmax_layer.h"
// #include "perf_cnt.h"
#include "snrt.h"
// #include "printf.h"
#include "utils.h"

int main() {
    // FIXME: We need to define the DRAM addresses for 
    // all data in the layer struct, because otherwise
    // it will be overwritten 
    softmax_l.ifmap = (float*)softmax_ifmap_dram;
    // softmax_l.result = (float*)softmax_ofmap_dram;

    // checksum = (float*)softmax_checksum;

    softmax_layer(&softmax_l);

    snrt_global_barrier();

    // uint32_t error = check_softmax_layer(&linear_l, (float*)linear_checksum);

    return 0;
}