# Applications

This subdirectory contains some applications or benchmarks specifically implemented and optimized for Snitch.

## Contents
- Data generation:
    - `datagen.py`: script to generate data and expected results for various benchmarks
    - `data`: output folder of `datagen.py` which also contains the configuration to generate the data
- `src`:
    - `kernels`: basic kernels, currently contains `GEMM`, `BatchNorm`, `Maxpool`, `Fusedconv`
    - `layers`: wraps the kernel to form a DNN layer. Manages data-movement, synchronization, double buffering etc.
    - `utils`: some helpful functions for benchmarking, verification, fast `memset`
    - `net_layer.c`: various ready tests to run layers.
- `include`: includes `layer` struct.

## SW Testbenches
There are currently a few tests for various layer types. Some additional information about these tests is given below:
- `net_maxpool.c`: Naive implementation of a maxpooling layer, not optimized in any way due to memory-boundness
- `net-batchnorm.c`: Implementation of a batchnorm layer with SSR streams (both read and write)
- `net-conv2d.c`: Implementation and tiling of a 2D convolution that can be distributed to multiple clusters. The convolution is implemented as an `im2col` transformation (performed by 2D DMA transfers) + optimized GEMM. The memory layout of input and output feature map is Height x Width x Channels. The convolution is globally parallelized over output channels. Inside a cluster, the output pixels are distributed among the cores. There is an option to load the feature map from a different cluster instead of the main memory by setting `cluster2cluster` in the layer struct to `1`. Currently only `fp64` is implemented, but the data movement for `fp32` or lower precision SIMD should be analogously.
- `net-gemm.c`: Testbench to benchmark the optimized GEMM implementation for different memory layouts, dimensions and precisions.
- `net-fusedconv.c`: Implementation of a fused kernel with Conv2d + BatchNorm + ReLU. The interface of the kernel is compatible with DORY. Parameters of a tile can be specified in `data/fusedconv_param.json`. Supported paramters are input/output dimension, padding, kernel dimension & stride, flags for BatchNorm and ReLU. Further there are two additional specialized kernels 1) a CHW kernel for input layers with very few input channels, the output of this kernel is in the HWC layout again 2) A depthwise kernel

## Usage
To run a specific benchmark, first configure the dimensions and the desired precision `data/app_params.json`.
```
{
    kernel: "GEMM"
    M: 16,
    N: 16,
    K: 16,
    alpha: 0,
    transpose_A: false,
    transpose_B: true,
    prec: 16
}
```

The file will be automatically generated with a `cmake` macro and is stored in `data/data_app.h`. The result will also be checked. Reference is a golden model written in `python` with help of the `torch`.

The applications are compiled into a folder which can be enabled by adding `add_subdirectory(${SNITCH_SOFTWARE_DIR}/applications` to `CMakeLists.txt` in the specific `sw` folder.

## Requirements
- `torch`

# Running ViT and GPT Models on the Snitch Cluster

## Introduction

This repository provides implementations of the Multi-Head Attention (MHA) and Multi-Layer Perceptron (MLP) layers for Vision Transformers (ViT) and Generative Pre-trained Transformer (GPT) models. The applications are designed to run on the Snitch cluster, leveraging its unique architecture for efficient execution.
This work stems from a journal paper currently under review at IEEE Transactions on Circuits and Systems for Artificial Intelligence. A preview of the paper can be found [here](https://arxiv.org/pdf/2405.19284).

The below figure shows a block diagram of the basic Attention layer. 

![Attention Layer](https://github.com/viv-eth/snitch_cluster/blob/llm/end-to-end/transformer_block.svg)

## This Work

In our work, we modified the basic Attention layer to include the following optimizations, as outlined in section V of our paper, titled "FOUNDATION MODEL LIBRARY":

- **FlashAttention-2 Algorithm**:
  We integrated the FlashAttention-2 algorithm into the Self-Attention mechanism. This enhancement significantly improves the computational efficiency by reducing memory usage and speeding up matrix multiplications, as depicted in the green section of the new topology diagram. For more details on the FlashAttention-2 algorithm, refer to the paper [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691).

- **Parallelization Techniques**:
  By parallelizing over the heads, we achieved substantial performance gains. This approach maximizes the utilization of the Snitch cluster's resources, allowing for more efficient processing of large-scale models.

- **FusedConcatLinear**:
  We introduced a fused operation that combines the head concatenation with the subsequent linear layer. This optimization, shown in the pink section of the diagram, reduces the computational overhead and speeds up the overall processing time.

- **LayerNorm and GELU Integration**:
  Our implementation ensures that LayerNorm and GELU activations are optimally placed within the pipeline. This careful placement enhances the stability and performance of the model.

- **Customized Hardware and Software Integration**:
  The Snitch cluster's unique architecture is leveraged to its full potential through these optimizations. We tailored both hardware and software components to work seamlessly together, ensuring efficient execution of the MHA and MLP layers.

The block diagram below illustrates the new topology we implemented to optimize the efficiency of running these layers on the Snitch cluster:

![New Topology](https://github.com/viv-eth/snitch_cluster/blob/llm/end-to-end/new_trafo_topology.png)

These enhancements collectively contribute to a more efficient and powerful implementation of the ViT and GPT models on the Snitch cluster, enabling faster training and inference times with lower resource consumption.

## Getting Started

### Installation

Starting from the root directory of the repository, run the following commands to build the applications:

```bash
cd snitch_cluster
git checkout llm/end-to-end
cd target/snitch_cluster
```

### Building Hardware
To build the hardware, navigate to the `target/snitch_cluster` directory and follow the instructions in the provided README file.
This will set up the necessary environment and compile the hardware of the Snitch cluster.

### Full Model Simulation (Slow)

To simulate the full ViT (encoder) or GPT (decoder) model, you can build the `encoder` and `decoder` application with the correct configuration file. 
The configuration files for the ViT model can be found in the `sw/dnn/encoder/data` directory, while the GPT model configuration files are located in the `sw/dnn/decoder/data` directory.
The prefixes of the subdirectories indicate the model architecture. Furthermore, we provide the configuration files for `FP32`, `FP16`, nd `FP8` precision. The following table summarizes the available configurations:


| Models| ViT-B | ViT-L | ViT-H   | GPT3-XL   | GPT-J  |
|------|-------|-------|---------|-----------|--------|
| Blocks | 12  | 24    | 32      | 40        | 28     |
| Params | 86M | 307M  | 632M    | 1.3B      | 6B     |
| E      | 768 | 1024  | 1280    | 2048      | 4096   |
| P      | 64  | 64    | 80      | 128       | 256    |
| S      | 197 | 197   | 197     | [128-2048]| [128-2048] |
| FF     | 3072| 4096  | 5120    | 8192      | 16384  |
| H      | 12  | 16    | 16      | 16        | 16     |

#### Repository Overview

Below figure gives an overview of the most important directories and files for building the encoder and decoder applications.

![Repository Info](https://github.com/viv-eth/snitch_cluster/blob/llm/end-to-end/repo_llm.png)

#### Building the Software

The default configuration from the `params.json` can be overwritten by setting the `DATA_CFG` environment variable. An example command to run the ViT-B model in `FP16` precision is shown below:

```bash
make DEBUG=ON DATA_CFG=sw/dnn/encoder/data/vit-b/vit-b-fp16.json sw/apps/dnn/encoder
```

This will build the binary that will run on the Snitch cluster target and generate the data for the model. If you wish to re-generate the data independently, you can use the provided Python script `datagen.py`. This script generates the input data and expected results for the specified model and configuration. The generated data is stored in the `data` directory of the respective model. We provided the configurations for the benchmarked models in the `encoder/data` and `decoder/data` directories.

The command for data generation is as follows:

```bash
python sw/dnn/decoder/scripts/datagen.py -c sw/dnn/<app_name>/data/<model>/<model-fpXX>.json --section="" sw/dnn/<app_name>/data/data.h
```

This will generate the data header file `data.h` in the specified output directory `sw/dnn/<app_name>/data/`.

#### Running the Application

After building the software, you can run the applications on the Snitch cluster. Below is an example command using the `QuestaSim` simulator: 

```bash
bin/snitch_cluster.vsim sw/apps/dnn/<app_name>/build/<app_name>.elf
```

### Single Layer Simulation (Fast)

You can follow the above instructions to build the software applications. This will build all of the `dnn` applications, including the MHA and MLP layers for ViT and GPT models.
If you prefer to only build the MHA and MLP layers, you can run the following commands:

```bash
make DEBUG=ON sw/apps/dnn/<app_name>
```

The parameters of the MHA and MLP layers can be configured in the `data/params.json` file.  The current configuration will run a single tile of the MHA and MLP computation.
