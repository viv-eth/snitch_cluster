// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Viviane Potocnik <vivianep@iis.ee.ethz.ch>

#include <math.h>
#include <stdint.h>

#include "blas.h"
#include "dnn.h"
#include "data.h"

#include "snrt.h"

int main() {
    mlp_layer(&mlp_args);
    return 0;
}