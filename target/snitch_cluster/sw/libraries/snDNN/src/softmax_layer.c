// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "softmax_layer.h"

#include "layer.h"
#include "softmax.h"
// #include "printf.h"
// #include "sndnn.h"
#include "snrt.h"

void softmax_layer(softmax_layer_t *const l) {
    uint32_t cluster_num = snrt_cluster_num();
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t compute_num = snrt_cluster_compute_core_num();
    uint32_t compute_id = snrt_global_core_idx();

    uint32_t ifmap_size =
        l->BATCH_SIZE * l->SEQ_LEN * l->INPUT_SAMPLES * sizeof(float);
    uint32_t ofmap_size = ifmap_size;

    void *ptr = (float *)snrt_l1_next();
    float *ifmap = ptr;
    ptr += ifmap_size;
    float *ofmap = ptr;
    ptr += ofmap_size;
    // float *result = ptr;
    // ptr += ofmap_size;

    if (compute_id == 0) {
        // Determine the memory usage of the cluster
        // in kB
        uint32_t mem_usage = (ifmap_size + 1 * ofmap_size) / 1024;
        printf("Softmax layer: Memory usage: %d kB\n", mem_usage);
    }

    // DMA transfer the ifmap into the cluster TCDM
    // and the golden model ofmap
    // 2D DMA transfer: dst, src, size, dst_stride, src_stride, repetitions
    if (snrt_is_dm_core()) {
        // printf("DMA transfer start...\n");
        // ifmap dimensions: BATCH_SIZE x SEQ_LEN x INPUT_SAMPLES
        snrt_dma_txid_t txid_ifmap = snrt_dma_start_2d(
                ifmap,                                          /* dst */
                l->ifmap,                                       /* src */
                l->BATCH_SIZE * sizeof(float),                  /* size */
                l->BATCH_SIZE * sizeof(float),                  /* dst stride */
                l->BATCH_SIZE * sizeof(float),                  /* src stride */
                l->SEQ_LEN * l->INPUT_SAMPLES * sizeof(float)); /* repetitions */

        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()) {
        // determine the row offset for each core
        int32_t row_offset = compute_id * l->INPUT_SAMPLES;

        // determine the row stride of each matrix
        int32_t ldI = compute_num * l->INPUT_SAMPLES;

        // determine the batch offset for each core
        int32_t batch_offset = l->SEQ_LEN * l->INPUT_SAMPLES;

        // printf("row_offset: %d, ldI: %d\n", row_offset, ldI);
        // printf("address of ifmap: %p\n", ifmap);
        // printf("address of ifmap at row_offset: %p\n", ifmap + row_offset);
        // printf("ifmap[%d] = %x\n", row_offset, ifmap[row_offset]);
        // printf("Compute core %d/%d\n", compute_id, compute_num);
        benchmark_get_cycle();
        softmax_fp32(&ifmap[row_offset], &ofmap[row_offset], ldI, batch_offset,
                     l->BATCH_SIZE, l->SEQ_LEN / compute_num, l->INPUT_SAMPLES);
        benchmark_get_cycle();


        // if (compute_id == 0) {
        //     // Compare the ofmap with the golden model
        //     uint32_t error = 0;
        //     int32_t batch_size = l->BATCH_SIZE;
        //     int32_t seq_len = l->SEQ_LEN;
        //     int32_t input_samples = l->INPUT_SAMPLES;

        //     // for (int32_t b = 0; b < batch_size; b++) {
        //     //     for (int32_t s = 0; s < seq_len; s++) {
        //     //         for (int32_t i = 0; i < input_samples; i++) {
        //     //             int32_t idx = b * batch_offset + s * ldI + i;
        //     //             if (fabs(ofmap[idx] - result[idx]) > 0.0001) {
        //     //                 error++;
        //     //                 printf("ofmap[%d] = %f, result[%d] = %f\n", idx,
        //     //                        ofmap[idx], idx, result[idx]);
        //     //             }
        //     //         }
        //     //     }
        //     // }

        //     printf("Softmax layer: %s\n", error ? "FAILED" : "PASSED");
        //     if (error) {
        //         printf("[%d]/[%d] errors\n", error, batch_size * seq_len *
        //                                            input_samples);
        //     }

        // }

        // snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();

    } else {
        // snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();
    }

    snrt_global_barrier();
}