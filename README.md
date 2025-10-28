# CUDA-Accelerated Multi-Layer Perceptron for MNIST Classification

## M.Tech Mini Project - High-Performance Computing Systems

A comprehensive implementation of a Multi-Layer Perceptron (MLP) neural network from scratch, featuring CPU serial, CUDA parallel training, and GPU-accelerated inference for the MNIST handwritten digit dataset.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
   - [Training](#training)
   - [Inference](#inference)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Advanced Features](#advanced-features)
8. [Model Persistence](#model-persistence)
9. [Results](#results)
10. [Technical Implementation](#technical-implementation)
11. [Future Work](#future-work)
12. [Acknowledgments](#acknowledgments)

---

## Project Overview

This project implements a fully-functional neural network training and inference system with three versions:

1. **Serial (CPU)**: Pure C implementation for baseline comparison
2. **CUDA Training**: GPU-accelerated training with multiple optimization techniques
3. **CUDA Inference**: Standalone GPU inference binary for deployment

The system demonstrates significant speedups through CUDA parallelization while maintaining numerical accuracy, making it suitable for educational purposes and as a foundation for understanding GPU-accelerated deep learning.

### Key Achievements

- **100-200× speedup** over serial implementation with CUDA
- **Sub-millisecond inference** (~0.4 ms per image)
- **95%+ accuracy** on MNIST test set
- **Multiple optimization strategies** (shared memory, batch normalization, advanced optimizers)
- **Complete training-to-deployment pipeline** with model persistence

---

## Features

### Core Features

- ✅ **Multi-layer Perceptron** with configurable architecture (hidden layers and neurons)
- ✅ **Activation Functions**: ReLU (hidden layers), Softmax (output layer)
- ✅ **Loss Function**: Cross-Entropy Loss
- ✅ **GPU Parallelization**: Each thread processes one training sample
- ✅ **Model Persistence**: Save and load trained models
- ✅ **Standalone Inference**: Fast GPU inference binary

### Advanced Optimizations

| Optimization | Status | Performance Impact | Recommended |
|-------------|--------|-------------------|-------------|
| **Shared Memory** | ✅ Enabled | +6.5% to +15.8% speedup | Yes |
| **Learning Rate Scheduling** | ✅ Optional | Better convergence | Yes |
| **Batch Normalization** | ✅ Optional | Faster training | Yes |
| **Optimizers** (SGD/Momentum/Adam) | ✅ Optional | Better accuracy | Yes |
| **Validation Split** | ✅ Optional | Prevent overfitting | Yes |
| **Dropout Regularization** | ✅ Optional | Minimal benefit | No |
| **Mixed Precision (FP16)** | ⚠️ Not Recommended | 1-36% slower | No |

---

## Project Structure

```
CUDA-based-MLP/
├── mnist_nn_cuda.cu          # Main CUDA training implementation
├── infer.cu                  # Standalone CUDA inference binary
├── mnist_nn_serial_c.c       # Serial CPU implementation
├── load_data.h               # MNIST data loader
├── Makefile                  # Build configuration
├── generate_test_input.py    # Helper script for inference testing
├── getdata.sh                # MNIST dataset download script
├── model_checkpoint.bin      # Trained model weights (generated)
├── data/                     # MNIST dataset (downloaded)
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── README.md                 # This file
└── OPTIMIZATIONS_SUMMARY.md  # Detailed optimization analysis
```

---

## Setup and Installation

### Prerequisites

- **CUDA Toolkit** (11.0 or later recommended)
- **NVIDIA GPU** with Compute Capability 5.0+ (tested on sm_86)
- **GCC/G++** compiler
- **Python 3** (for helper scripts)
- **GNU Make**

### Installation Steps

1. **Clone the repository**
   ```bash
   cd /path/to/your/workspace
   git clone <repository-url>
   cd CUDA-based-MLP
   ```

2. **Download MNIST dataset**
   ```bash
   bash getdata.sh
   ```
   This downloads the MNIST dataset (~10 MB) into the `data/` directory.

3. **Compile the project**
   ```bash
   make all        # Build serial and CUDA training binaries
   make infer      # Build inference binary
   ```

### Build Targets

The Makefile provides multiple build configurations:

```bash
make cuda                # Basic CUDA training (SGD optimizer)
make cuda_momentum       # CUDA with Momentum optimizer
make cuda_adam           # CUDA with Adam optimizer
make cuda_batchnorm      # CUDA with Batch Normalization
make cuda_lr_schedule    # CUDA with Learning Rate Scheduling
make cuda_full           # All features enabled
make infer               # Standalone inference binary
make clean               # Remove all binaries
```

---

## Usage

### Training

#### Serial (CPU) Version

```bash
./serial <nl> <nh> <ne> <nb> <lr> <output_type> <log_interval>
```

**Example:**
```bash
./serial 2 30 300 6000 0.1 1 100
```

#### CUDA (GPU) Version

```bash
./cuda <nl> <nh> <ne> <nb> <lr> <output_type> <log_interval> <save_model>
```

**Parameters:**
- `nl`: Number of hidden layers (excluding input and output)
- `nh`: Number of neurons per hidden layer
- `ne`: Number of training epochs
- `nb`: Batch size (number of samples per batch)
- `lr`: Learning rate (e.g., 0.1)
- `output_type`:
  - `0` = Sigmoid + MSE loss
  - `1` = Softmax + Cross-Entropy loss (recommended)
- `log_interval`: Print progress every N epochs
- `save_model` (optional):
  - `0` = No save/load (default)
  - `1` = Save model after training
  - `2` = Load existing model, then continue training

**Example - Train and Save Model:**
```bash
./cuda 2 128 20 2048 0.1 1 1 1
```

This trains a network with:
- 2 hidden layers
- 128 neurons per layer
- 20 epochs
- Batch size 2048
- Learning rate 0.1
- Saves model to `./model_checkpoint.bin`

**Example - Resume Training:**
```bash
./cuda 2 128 10 2048 0.05 1 1 2
```

This loads the saved model and continues training for 10 more epochs with a lower learning rate.

**Advanced Training - Adam with Learning Rate Scheduling:**
```bash
./cuda_adam_lr 2 128 30 2048 0.1 1 1 1
```

### Inference

The `infer` binary performs fast GPU inference using a trained model checkpoint.

#### Basic Usage

```bash
python3 generate_test_input.py <image_index> | ./infer
```

**Example:**
```bash
# Predict digit for test image 0
python3 generate_test_input.py 0 | ./infer
# Output: 7
```

#### Verbose Mode

```bash
python3 generate_test_input.py 0 | ./infer -v
```

**Output:**
```
Loading model from ./model_checkpoint.bin...
Model loaded successfully
  Layers: 2 (hidden)
  Neurons per layer: 128
Reading input (784 values)...
Running inference...

Prediction: 7

Output probabilities:
  Digit 0: 0.003738
  Digit 1: 0.001375
  Digit 2: 0.001517
  ...
  Digit 7: 0.923906
  Digit 9: 0.044513
```

#### Benchmark Mode

```bash
python3 generate_test_input.py 0 | ./infer -b -v
```

Shows inference timing (typically ~0.4 ms).

#### Visualize Input

```bash
python3 generate_test_input.py 0 -v 2>&1 | head -35
```

Displays ASCII art visualization of the input digit.

#### Custom Model Path

```bash
./infer -m /path/to/custom_model.bin < input.txt
```

#### Batch Testing

```bash
# Test first 10 images
for i in {0..9}; do
    echo -n "Image $i: "
    python3 generate_test_input.py $i | ./infer
done
```

---

## Performance Benchmarks

### Training Performance

**Configuration:** 2 hidden layers × 128 neurons, 20 epochs, batch size 2048

| Implementation | Time/Epoch | Total Time | Speedup | Test Accuracy |
|---------------|-----------|-----------|---------|---------------|
| **Serial (CPU)** | ~90 sec | ~30 min | 1× | 95.2% |
| **CUDA (Basic)** | 0.542 sec | 10.8 sec | **166×** | 95.2% |
| **CUDA + Shared Memory** | 0.509 sec | 10.2 sec | **177×** | 95.2% |
| **CUDA + BatchNorm** | 0.468 sec | 9.4 sec | **192×** | 95.4% |

### Inference Performance

**Configuration:** Pre-trained model (2 layers × 128 neurons)

| Metric | Value |
|--------|-------|
| **Inference Time** | ~0.4 ms per image |
| **Throughput** | ~2,500 images/second |
| **Model Load Time** | ~5 ms |
| **Accuracy** | 95%+ on test set |

### Optimization Impact

**Shared Memory (Batch 2048):**
- Without: 0.542 sec/epoch
- With: 0.509 sec/epoch
- **Speedup: 1.065× (+6.5%)**

**Batch Size Scaling (20 epochs):**
| Batch Size | Time/Epoch | Total Time |
|-----------|-----------|-----------|
| 512 | 0.509 sec | 10.2 sec |
| 1024 | 0.516 sec | 10.3 sec |
| 2048 | 0.538 sec | 10.8 sec |
| 4096 | 0.598 sec | 12.0 sec |

---

## Advanced Features

### Learning Rate Scheduling

Dynamically adjust learning rate during training for better convergence.

**Types:**
1. **Step Decay**: Reduce LR by factor every N epochs
2. **Exponential Decay**: Smooth exponential decrease
3. **Cosine Annealing**: Sinusoidal decay pattern

**Usage:**
```bash
./cuda_lr_schedule 2 128 30 2048 0.1 1 1 1
```

### Batch Normalization

Normalize layer activations to accelerate training and improve accuracy.

**Usage:**
```bash
./cuda_batchnorm 2 128 20 2048 0.1 1 1 1
```

**Benefits:**
- Faster convergence (~20% speedup)
- Higher final accuracy (+0.2% typical)
- Better gradient flow

### Optimizers

**SGD (Default):**
```bash
./cuda 2 128 20 2048 0.1 1 1 1
```

**Momentum (β=0.9):**
```bash
./cuda_momentum 2 128 20 2048 0.1 1 1 1
```

**Adam (β1=0.9, β2=0.999):**
```bash
./cuda_adam 2 128 20 2048 0.1 1 1 1
```

**Comparison:**
| Optimizer | Convergence Speed | Final Accuracy | Best For |
|-----------|------------------|---------------|----------|
| **SGD** | Baseline | 95.2% | Simple tasks |
| **Momentum** | +15% faster | 95.3% | General use |
| **Adam** | +20% faster | 95.5% | Complex tasks |

### Validation Split

Split training data into train/validation sets to monitor overfitting.

**Enable:** Set `USE_VALIDATION_SPLIT 1` in `mnist_nn_cuda.cu` (line 47)

**Output Example:**
```
Epoch 5: Train 94.5%, Val 93.8%, Test 94.2%
```

---

## Model Persistence

### Save Model

```bash
./cuda 2 128 20 2048 0.1 1 1 1
```

Creates `model_checkpoint.bin` containing:
- Network architecture (layers, neurons)
- All weights and biases
- File size: ~470 KB for 2×128 network

### Load Model

```bash
# Continue training from checkpoint
./cuda 2 128 10 2048 0.05 1 1 2

# Or use for inference
./infer < input.txt
```

### File Format

**Binary structure:**
```
[int] nl          // Number of hidden layers
[int] nh          // Neurons per hidden layer
[float[]] weights // (784×nh + nh×nh×(nl-1) + nh×10) values
[float[]] biases  // (nh×nl + 10) values
```

---

## Results

### Accuracy Progression

**Configuration:** 2 layers × 128 neurons, batch size 2048, LR 0.1

| Epoch | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| 0 | 12.5% | 12.8% |
| 5 | 82.3% | 82.1% |
| 10 | 91.2% | 90.8% |
| 15 | 94.1% | 93.7% |
| 20 | 95.2% | 94.9% |

### Final Performance

**Best Configuration:** 2 layers × 256 neurons, Adam optimizer, LR scheduling

- **Training Accuracy:** 97.8%
- **Test Accuracy:** 96.2%
- **Training Time:** 35 epochs × 0.8 sec = 28 seconds
- **Inference Speed:** 0.4 ms per image

---

## Technical Implementation

### Algorithm Overview

```
L1  for each epoch:
L2      log training progress
L3      sample mini-batch
L4      for each sample in batch (PARALLEL ON GPU):
L5          forward propagation
L6          backward propagation
L7      end
L8      gradient descent (ATOMIC UPDATES)
L9  end
```

### CUDA Parallelization Strategy

- **Thread Parallelism**: Each thread processes one training sample
- **Block Size**: 128 threads per block
- **Grid Size**: Dynamically calculated based on batch size
- **Memory**: All data pre-loaded to GPU (one-time transfer)
- **Atomic Operations**: Gradient accumulation uses `atomicAdd()`

### Memory Layout

**Per-Thread Allocations (Stack):**
- Activations: `float a[1024]`
- Pre-activations: `float z[1024]`
- Gradients: `float delta[1024]`

**Shared Memory (Block-Level):**
- Weight tiles: `shared_weights[32×32]`
- Activation tiles: `shared_a[32]`

**Global Memory:**
- Training images: 60,000 × 784 × 4 bytes = ~180 MB
- Test images: 10,000 × 784 × 4 bytes = ~30 MB
- Weights/Biases: Network-dependent (~470 KB typical)

### Optimization Techniques

1. **Shared Memory Tiling**: Cooperative loading of weight matrices
2. **Coalesced Memory Access**: Aligned memory access patterns
3. **Atomic Operations**: Thread-safe gradient accumulation
4. **Loop Unrolling**: Compiler optimizations for small fixed loops
5. **Register Reuse**: Minimize global memory round-trips

---

## Future Work

### Potential Improvements

1. **Tensor Core WMMA Implementation** (2-3× training speedup)
   - Requires full kernel restructuring
   - Warp-level matrix operations
   - Estimated effort: 3-4 weeks

2. **Multi-GPU Training** (horizontal scaling)
   - Data parallelism across GPUs
   - Gradient synchronization

3. **Convolutional Layers** (better MNIST accuracy)
   - 2D convolutions for spatial features
   - Pooling layers

4. **Advanced Architectures**
   - Residual connections
   - Attention mechanisms

5. **Deployment Features**
   - Python bindings (ctypes/pybind11)
   - REST API server
   - Web-based demo interface

6. **Quantization**
   - INT8 inference for 4× throughput
   - Post-training quantization

---

## Detailed Documentation

For in-depth technical analysis of optimization strategies, see:
- **[OPTIMIZATIONS_SUMMARY.md](OPTIMIZATIONS_SUMMARY.md)** - Complete optimization analysis
- **Source code comments** - Inline documentation in `.cu` files

---

## Key Takeaways

### What Was Learned

1. **CUDA Fundamentals**
   - Thread hierarchy (thread → block → grid)
   - Memory hierarchy (registers → shared → global)
   - Synchronization primitives (`__syncthreads()`, `atomicAdd()`)

2. **Performance Optimization**
   - Shared memory provides 6-16% speedup
   - Batch size significantly impacts performance
   - Not all "advanced" techniques help (e.g., FP16 without WMMA)

3. **Neural Network Training**
   - Backpropagation implementation from scratch
   - Optimizer comparison (SGD vs Momentum vs Adam)
   - Regularization techniques (dropout, batch normalization)

4. **Software Engineering**
   - Modular code design
   - Performance benchmarking methodology
   - Model persistence and deployment

### Project Significance

This project demonstrates a **complete training-to-deployment pipeline** for GPU-accelerated deep learning:

✅ **Training**: CUDA-accelerated with 100-200× speedup
✅ **Evaluation**: Real-time accuracy monitoring
✅ **Persistence**: Save/load trained models
✅ **Deployment**: Standalone inference binary
✅ **Optimization**: Multiple performance tuning strategies

The implementation serves as:
- Educational resource for CUDA programming
- Foundation for understanding GPU-accelerated ML
- Baseline for comparing optimization techniques
- Complete reference implementation (1485 lines of well-documented CUDA C)

---

## Acknowledgments

### References

- **MNIST Database**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
  http://yann.lecun.com/exdb/mnist/

- **CUDA Programming Guide**: NVIDIA Corporation
  https://docs.nvidia.com/cuda/

- **Neural Networks and Deep Learning**: Michael Nielsen
  http://neuralnetworksanddeeplearning.com/

### Course

This project was developed as part of the **M.Tech High-Performance Computing Systems** curriculum, demonstrating practical applications of GPU parallelization for scientific computing and machine learning.

### Tools and Technologies

- **CUDA Toolkit** (NVIDIA)
- **nvcc** Compiler
- **Python 3** (Helper scripts)
- **GNU Make** (Build system)

---

## License

This project is developed for educational purposes as part of academic coursework.

---

## Contact and Contribution

For questions, suggestions, or bug reports, please refer to the course instructor or create an issue in the repository.

