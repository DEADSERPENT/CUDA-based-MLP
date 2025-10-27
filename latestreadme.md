# CUDA-based Multi-Layer Perceptron (MLP) for MNIST

A high-performance implementation of a feedforward neural network for MNIST digit classification, with both CPU (serial) and GPU (CUDA) versions for performance comparison.

## Overview

This project implements a fully-connected neural network from scratch in C/CUDA for classifying handwritten digits from the MNIST dataset. It features:

- **Serial (CPU) version** in pure C
- **Parallel (GPU) version** using CUDA
- Flexible architecture with configurable layers and neurons
- Multiple activation functions (ReLU, Sigmoid, Softmax)
- Mini-batch gradient descent optimization
- Cross-entropy and MSE loss functions

## Features

- **Flexible Network Architecture**: Configure number of hidden layers and neurons per layer
- **Activation Functions**:
  - ReLU for hidden layers
  - Softmax for output layer (with cross-entropy loss)
  - Sigmoid alternative (with MSE loss)
- **Mini-batch Training**: Efficient stochastic gradient descent with configurable batch sizes
- **GPU Acceleration**: CUDA implementation for parallel training
- **Performance Metrics**: Track training/test accuracy and timing

## Requirements

### Hardware
- CPU: Any modern x86-64 processor
- GPU (for CUDA version): NVIDIA GPU with compute capability 8.6+ (RTX 3060 or higher recommended)

### Software
- GCC compiler
- NVIDIA CUDA Toolkit (nvcc)
- wget (for downloading dataset)
- gunzip (for extracting dataset)

## Installation

### 1. Clone or Download the Project

```bash
cd /path/to/project
```

### 2. Download MNIST Dataset

```bash
chmod +x getdata.sh
./getdata.sh
```

This will download and extract:
- 60,000 training images (28×28 pixels)
- 60,000 training labels
- 10,000 test images
- 10,000 test labels

### 3. Compile the Programs

```bash
make all
```

This compiles both versions:
- `serial` - CPU version
- `cuda` - GPU version

Or compile individually:
```bash
make serial    # CPU version only
make cuda      # GPU version only
```

## Usage

### Command Syntax

Both programs use the same command-line arguments:

```bash
./serial <layers> <neurons> <epochs> <batch_size> <learning_rate> <output_type> [log_interval]
./cuda   <layers> <neurons> <epochs> <batch_size> <learning_rate> <output_type> [log_interval]
```

### Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `layers` | Number of hidden layers (max 20) | 2, 3, 4 |
| `neurons` | Neurons per hidden layer | 30, 64, 128, 256 |
| `epochs` | Training epochs | 10, 30, 50 |
| `batch_size` | Mini-batch size | 32, 64, 128 |
| `learning_rate` | Learning rate (alpha) | 0.1, 0.3, 0.5 |
| `output_type` | 0: Sigmoid+MSE, 1: Softmax+CrossEntropy | 0 or 1 |
| `log_interval` | Print progress every N epochs (optional, default=1) | 1, 5, 10 |

### Example Commands

**Quick test (5 epochs):**
```bash
./serial 2 128 5 64 0.5 1 1
```

**Standard training (30 epochs):**
```bash
./serial 2 128 30 64 0.5 1 5
./cuda 2 128 30 64 0.5 1 5
```

**Deep network (4 layers):**
```bash
./cuda 4 256 50 64 0.3 1 10
```

## Project Structure

```
CUDA-based-MLP/
├── mnist_nn_serial_c.c    # CPU implementation
├── mnist_nn_cuda.cu       # GPU implementation
├── load_data.h            # MNIST data loader
├── Makefile               # Build configuration
├── getdata.sh             # Dataset download script
├── data/                  # MNIST dataset directory
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── serial                 # Compiled CPU executable
├── cuda                   # Compiled GPU executable
└── latestreadme.md        # This file
```

## Performance Comparison

### Configuration
- Network: 2 hidden layers, 128 neurons each
- Training: 30 epochs, batch size 64, learning rate 0.5
- Output: Softmax + Cross-Entropy

### Results

| Version | Test Accuracy | Time per Epoch | Hardware |
|---------|---------------|----------------|----------|
| Serial (CPU) | 77.68% | ~0.057s | x86-64 CPU |
| CUDA (GPU) | 71.86% | ~0.434s | RTX 3060 |

**Note:** The CUDA version is slower for this small network size due to overhead. GPU acceleration shows benefits with larger networks (more layers/neurons) or larger batch sizes.

## Technical Details

### Neural Network Architecture

**Input Layer:** 784 neurons (28×28 pixels flattened)

**Hidden Layers:** Configurable count and size
- Activation: ReLU (Rectified Linear Unit)
- Initialization: He initialization (Xavier variant)

**Output Layer:** 10 neurons (digits 0-9)
- Activation: Softmax (for classification probabilities)
- Loss: Cross-Entropy

### Training Algorithm

1. **Forward Propagation**: Compute activations through all layers
2. **Backward Propagation**: Calculate gradients using chain rule
3. **Mini-batch Gradient Descent**: Update weights/biases using accumulated gradients
4. **Evaluation**: Periodic accuracy checks on training and test sets

### CUDA Implementation Details

- **Thread Organization**: 128 threads per block
- **Parallelization**: Each thread processes one training sample in the mini-batch
- **Memory**: Separate device memory for weights, biases, activations, and deltas
- **Synchronization**: Atomic operations for weight updates
- **Compute Architecture**: Optimized for sm_86 (RTX 3060)

## Optimization Tips

### For Better Accuracy:
1. Increase epochs: 50-100
2. Try different learning rates: 0.1, 0.3, 0.5
3. Adjust network size: More neurons (256, 512) or layers (3-4)
4. Always use Softmax + Cross-Entropy for classification

### For Better GPU Performance:
1. Increase batch size: 128, 256, 512
2. Use larger networks: 4+ layers, 256+ neurons
3. Ensure CUDA architecture matches your GPU (`-arch sm_XX`)

## Common Issues & Solutions

### Data Loading Errors
```
Error: couldn't open image file
```
**Solution:** Run `./getdata.sh` to download the dataset

### CUDA Out of Memory
```
CUDA error: out of memory
```
**Solution:** Reduce batch size or network size

### Low Accuracy (<20%)
**Solution:**
- Increase epochs
- Adjust learning rate (try 0.3 or 0.5)
- Ensure output_type=1 (Softmax+CrossEntropy)

## Build System

### Makefile Targets

```bash
make all       # Build both versions
make serial    # Build CPU version only
make cuda      # Build GPU version only
make clean     # Remove executables
make re        # Clean and rebuild all
```

### Compiler Flags

**GCC (Serial):**
- `-g`: Debug symbols
- `-lm`: Math library

**NVCC (CUDA):**
- `-arch sm_86`: Target RTX 3060 architecture
- `-g -G`: Debug symbols for host and device code

## Code Structure

### Serial Version (mnist_nn_serial_c.c)

Key functions:
- `forward_prop()`: Forward propagation through network
- `forward_back_prop()`: Combined forward and backward pass
- `weight_x_a()`: Matrix-vector multiplication for forward pass
- `weight_x_d()`: Matrix-vector multiplication for backward pass
- `evaluate()`: Calculate accuracy on dataset
- `relu()` / `relu_prime()`: Activation functions
- `softmax()`: Output layer activation
- `delta_cross_entropy()`: Cross-entropy gradient

### CUDA Version (mnist_nn_cuda.cu)

Key components:
- `__global__ one_learning_cycle()`: Main CUDA kernel for training
- `__device__` functions: GPU-optimized math operations
- Host functions: Data management and evaluation
- Memory transfers: Efficient CPU-GPU data movement

### Data Loader (load_data.h)

- `load_mnist()`: Main loading function
- `read_mnist_char_image()`: Read image files
- `read_mnist_char_label()`: Read label files
- `FlipLong()`: Byte order conversion (big-endian to little-endian)
- `image_char2float()`: Normalize pixel values to [0,1]

## Future Improvements

- [ ] Implement dropout for regularization
- [ ] Add momentum/Adam optimizer
- [ ] Support for larger networks (>20 layers)
- [ ] Data augmentation
- [ ] Model saving/loading
- [ ] Validation set split
- [ ] Learning rate scheduling
- [ ] Batch normalization
- [ ] GPU optimization for small batch sizes

## References

- MNIST Dataset: [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)
- He Initialization: [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
- CUDA Programming Guide: [NVIDIA Documentation](https://docs.nvidia.com/cuda/)

## License

Educational project for learning CUDA and neural network implementation.

## Author

Created as a mini-project for CUDA-based parallel computing and deep learning fundamentals.

---

**Last Updated:** October 2025
**CUDA Version:** 13.0
**Tested On:** RTX 3060, Ubuntu 22.04 (WSL2)
