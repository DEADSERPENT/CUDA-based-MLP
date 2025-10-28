COMPILER1 = gcc
CFLAGS1 = -g -lm

COMPILER2 = nvcc
CFLAGS2 = -arch sm_86 -g -G

all: serial cuda

cuda: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda $(CFLAGS2)

# Inference binary for CUDA model
infer: infer.cu
	$(COMPILER2) infer.cu -o infer $(CFLAGS2)

# Build with Momentum optimizer (OPTIMIZER_TYPE=1)
cuda_momentum: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_momentum $(CFLAGS2) -DOPTIMIZER_TYPE=1

# Build with Adam optimizer (OPTIMIZER_TYPE=2)
cuda_adam: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_adam $(CFLAGS2) -DOPTIMIZER_TYPE=2

# Build with Learning Rate Scheduling enabled (Exponential Decay)
cuda_lr_schedule: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_lr_schedule $(CFLAGS2) -DUSE_LR_SCHEDULE=1

# Build with Adam + LR Scheduling
cuda_adam_lr: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_adam_lr $(CFLAGS2) -DOPTIMIZER_TYPE=2 -DUSE_LR_SCHEDULE=1

# Build with Batch Normalization enabled
cuda_batchnorm: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_batchnorm $(CFLAGS2) -DUSE_BATCH_NORM=1

# Build with LR Scheduling + Batch Normalization
cuda_lr_bn: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_lr_bn $(CFLAGS2) -DUSE_LR_SCHEDULE=1 -DUSE_BATCH_NORM=1

# Build with Adam + LR Scheduling + Batch Normalization (full features)
cuda_full: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda_full $(CFLAGS2) -DOPTIMIZER_TYPE=2 -DUSE_LR_SCHEDULE=1 -DUSE_BATCH_NORM=1

serial: mnist_nn_serial_c.c
	$(COMPILER1) mnist_nn_serial_c.c -o serial $(CFLAGS1)

clean:
	rm -f serial cuda cuda_momentum cuda_adam cuda_batchnorm cuda_lr_schedule cuda_adam_lr cuda_lr_bn cuda_full infer

re: clean all
