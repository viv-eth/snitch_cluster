#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>

import argparse
import pathlib
import json5
import sys
import os
import torch
import numpy as np
import re

import pyflexfloat as ff

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../util/sim/"))
import data_utils  # noqa: E402
from data_utils import emit_license, format_array_declaration, format_struct_definition, \
                       format_array_definition, format_ifdef_wrapper  # noqa: E402

torch.manual_seed(42)

# AXI splits bursts crossing 4KB address boundaries. To minimize
# the occurrence of these splits the data should be aligned to 4KB
BURST_ALIGNMENT = 4096

def infer_flg_implementation(flg_fp):
        # flg_fp: "fused_linear_gelu_fp64_opt"
        # create a regex with fp_<type>_<implementation>
        prec, impl = re.search(r'fused_linear_gelu_fp(\d+)_(\w+)', flg_fp).group(1, 2)
        return (int(prec) / 8), impl


def infer_gemm_implementation(gemm_fp):
    # gemm_fp: "gemm_fp64_opt"
    # create a regex with fp_<type>_<implementation>
    prec, impl = re.search(r'gemm_fp(\d+)_(\w+)', gemm_fp).group(1, 2)
    return (int(prec) / 8), impl


def sigmoid_gelu(x, x_shape, a, b):
    x = ff.np.float32(x)
    x = torch.tensor(x)
    result = torch.sign(x) * (a * (torch.clamp(torch.abs(x), max=-b) + b)**2 + 1)
    return ff.array(result.numpy(), x_shape)


def layernorm_golden_model(ifmap, eps):
    # Compute the mean and variance considering the last dimension (embeddings)
    # The dimensions for mean and variance calculations are (-1,), meaning the last dimension.
    # Keep the dimensions for broadcasting in normalization.
    mean = np.mean(ifmap, axis=(-1,), keepdims=True)
    diff = ifmap - mean
    var = np.mean(diff*diff, axis=(-1,), keepdims=True)

    # Normalize the input tensor
    ofmap = (ifmap - mean) / np.sqrt(var + eps)

    return ofmap


def flg_golden_model(self, alpha, a, b, beta, c, a_gelu, b_gelu):
        M, N, K = a.shape[0], b.shape[1], b.shape[0]
        result = beta * c
        result_shape = result.shape
        for m in range(M):
            for n in range(N):
                for k in range(K):
                    result[m][n] += a[m][k] * b[k][n]
                    result[m][n] = sigmoid_gelu(result[m][n], result_shape, a_gelu, b_gelu)
        return result


def gemm_exact_golden_model(alpha, a, b, beta, c):
    M, N, K = a.shape[0], b.shape[1], b.shape[0]
    result = beta * c
    for m in range(M):
        for n in range(N):
            for k in range(K):
                result[m][n] += a[m][k] * b[k][n]
    return result


def validate_config(**kwargs):
    """
    Validation of LayerNorm configuration parameters
    """
    # Aliases
    batch_size = kwargs['layernorm']['input_dim']['batch_size']
    seq_len = kwargs['layernorm']['input_dim']['seq_len']
    embeddings = kwargs['layernorm']['input_dim']['embeddings']

    # Calculate total TCDM occupation
    prec = data_utils.size_from_precision_t(kwargs['layernorm']['prec'])
    tiled_seq_len = seq_len / kwargs['layernorm']['n_tiles']
    total_size = batch_size * tiled_seq_len * embeddings * prec
    data_utils.validate_tcdm_footprint(total_size)

    assert kwargs['layernorm']['input_dim']['seq_len'] % kwargs['layernorm']['n_tiles'] == 0, 'Input dimension is not' \
        ' an integer multiple of' \
        ' tile size'
    assert kwargs['layernorm']['prec'] != "FP64", 'FP64 not supported'
    assert not (kwargs['layernorm']['implementation'] == "BASELINE"), 'No baseline implementations' \
        ' (switch to NAIVE)'
    assert not (kwargs['layernorm']['implementation'] ==
                "OPT_EX"), 'Expanding layernorm kernels not supported'
    assert not (kwargs['layernorm']['prec'] == "FP8" and kwargs['layernorm']['implementation'] == "NAIVE"), 'FP8 not ' \
        'supported in' \
        'naive ' \
        'implementation'

    """
    Validation of GEMM configuration parameters
    """
    M = kwargs['gemm']['M']
    N = kwargs['gemm']['N']
    K = kwargs['gemm']['K']
    m_tiles = kwargs['gemm']['m_tiles']
    n_tiles = kwargs['gemm']['n_tiles']
    k_tiles = kwargs['gemm']['k_tiles']
    parallelize_m = kwargs['gemm']['parallelize_m']
    parallelize_k = kwargs['gemm']['parallelize_k']
    transa = kwargs['gemm']['transa']
    transb = kwargs['gemm']['transb']
    beta = kwargs['gemm']['beta']

    frac_m = M / m_tiles
    frac_n = N / n_tiles
    frac_k = K / k_tiles

    dtype, impl = infer_gemm_implementation(kwargs['gemm']['gemm_fp'])

    prec = data_utils.size_from_precision_t(dtype)
    a_size = frac_m * frac_k * prec
    b_size = frac_k * frac_n * prec
    c_size = frac_m * frac_n * prec
    total_size = a_size
    total_size += b_size
    total_size += c_size
    data_utils.validate_tcdm_footprint(total_size)

    assert (M % m_tiles) == 0, 'M is not an integer multiple of tile size'
    assert (N % n_tiles) == 0, 'N is not an integer multiple of tile size'
    assert (K % k_tiles) == 0, 'K is not an integer multiple of tile size'
    assert not (parallelize_m and parallelize_k), 'Cannot parallelize K and M simultaneously'
    assert not transa, 'SIMD kernels don\'t support transposed A matrix'
    assert (dtype == 8) or (impl == 'baseline') or (impl == 'naive') \
        or transb, 'Optimized SIMD kernels only support transposed B matrix'
    assert not transb or n_tiles == 1, 'Tiling in the N dimension not supported' \
        ' if B is transposed'
    assert not transb or k_tiles == 1, 'Tiling in the K dimension not supported' \
        ' if B is transposed'
    assert (impl == 'baseline') or (impl == 'naive') or frac_n >= 8, \
        'N dimension of tile size must be greater or equal to the unrolling factor (8) ' \
        'when using optimized kernels'
    assert beta == 0 or beta == 1, 'Only values of 0 or 1 supported for beta'
    assert not (dtype == 8 and impl == "baseline"), 'No baseline implemented' \
        ' for FP64 (switch to NAIVE)'
    assert not (((dtype == 8) or (dtype == 4)) and impl == "opt_ex"), \
        'Expanding GEMM kernels' \
        ' not supported for FP64 and FP32'
    assert not (dtype == 1 and impl == "opt"), 'FP8 not supported in' \
        ' optimized implementation' \
        ' (switch to opt_ex)'

def emit_header(**kwargs):

    # Validate parameters
    validate_config(**kwargs)

    """
    LayerNorm data generation
    """
    batch_size = kwargs['layernorm']['input_dim']['batch_size']
    seq_len = kwargs['layernorm']['input_dim']['seq_len']
    embeddings = kwargs['layernorm']['input_dim']['embeddings']
    eps = kwargs['layernorm']['eps']
    prec = kwargs['layernorm']['prec']
    n_tiles = kwargs['layernorm']['n_tiles']
    implementation = kwargs['layernorm']['implementation']

    ff_desc = data_utils.ff_desc_from_precision_t(prec)
    ctype = data_utils.ctype_from_precision_t(prec)

    # Generate random input
    mlp_input = ff.array(np.random.rand(batch_size, seq_len, embeddings), ff_desc)
    x1_output = layernorm_golden_model(mlp_input, eps)

    mlp_input_uid = 'mlp_input'

    layernorm_cfg = {
        **kwargs['layernorm']['input_dim'],
        'n_tiles': n_tiles,
        'implementation': implementation,
        'eps': eps,
        'dtype': prec,
    }


    """
    FLG data generation
    """

    M, N, K = kwargs['fused_linear_gelu']['M'], kwargs['fused_linear_gelu']['N'], kwargs['fused_linear_gelu']['K']

    prec, _ = infer_flg_implementation(kwargs['fused_linear_gelu']['flg_fp'])

    ff_desc = data_utils.ff_desc_from_precision_t(prec)
    ctype = data_utils.ctype_from_precision_t(prec)

    W_ff = ff.array(np.random.rand(K, N), ff_desc)

    W_ff = W_ff.T if kwargs['fused_linear_gelu']['transb'] else W_ff

    W_ff_uid = 'W_ff'

    flg_cfg = {
        'prec': int(prec),
        **kwargs['fused_linear_gelu'],
    }

    W_ff = W_ff.flatten()


    """
    GEMM data generation
    """

    M, N, K = kwargs['gemm']['M'], kwargs['gemm']['N'], kwargs['gemm']['K']

    prec, _ = infer_gemm_implementation(kwargs['gemm']['gemm_fp'])

    ff_desc = data_utils.ff_desc_from_precision_t(prec)
    ctype = data_utils.ctype_from_precision_t(prec)

    W_mlp = ff.array(np.random.rand(K, N), ff_desc)

    W_mlp = W_mlp.T if kwargs['gemm']['transb'] else W_mlp

    W_mlp_uid = 'W_mlp'

    gemm_cfg = {
        'prec': int(prec),
        **kwargs['gemm'],
    }

    W_mlp = W_mlp.flatten()

    """
    MLP Configuration
    """
    mlp_args_cfg = {
        'layernorm_cfg': '&layernorm_cfg',
        'flg_cfg': '&flg_cfg',
        'gemm_cfg': '&gemm_cfg',
        'mlp_input': mlp_input_uid,
        'W_ff': W_ff_uid,
        'W_mlp': W_mlp_uid,
    }


    """
    Emit header file
    """
    data_str = [emit_license()]
    data_str += [format_array_declaration(ctype, mlp_input_uid, mlp_input.shape,
                 alignment=BURST_ALIGNMENT)]
    data_str += [format_array_declaration(ctype, W_ff_uid, W_ff.shape,
                 alignment=BURST_ALIGNMENT)]
    data_str += [format_array_declaration(ctype, W_mlp_uid, W_mlp.shape,
                    alignment=BURST_ALIGNMENT)]

    """
    Emit struct definitions
    """
    data_str += [format_struct_definition('layernorm_layer_t', 'layernorm_cfg', layernorm_cfg)]
    data_str += [format_struct_definition('flg_args_t', 'flg_cfg', flg_cfg)]
    data_str += [format_struct_definition('gemm_args_t', 'gemm_cfg', gemm_cfg)]
    data_str += [format_struct_definition('mlp_args_t', 'mlp_args', mlp_args_cfg)]

    data_str += [format_array_definition(ctype, mlp_input_uid, mlp_input,
                 alignment=BURST_ALIGNMENT)]
    data_str += [format_array_definition(ctype, W_ff_uid, W_ff,
                    alignment=BURST_ALIGNMENT)]
    data_str += [format_array_definition(ctype, W_mlp_uid, W_mlp,
                    alignment=BURST_ALIGNMENT)]

    data_str = '\n\n'.join(data_str)
    return data_str

def main():

    parser = argparse.ArgumentParser(description='Generate data for layernorm kernel')
    parser.add_argument(
        "-c", "--cfg",
        type=pathlib.Path,
        required=True,
        help='Select param config file kernel'
    )
    parser.add_argument(
        '--section',
        type=str,
        help='Section to store matrices in')
    parser.add_argument(
        'output',
        type=pathlib.Path,
        help='Path of the output header file')
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = json5.loads(f.read())
    param['section'] = args.section

    # Emit header file
    with open(args.output, 'w') as f:
        f.write(emit_header(**param))


if __name__ == '__main__':
    main()