COMPILER1 = gcc
CFLAGS1 = -g -lm

COMPILER2 = nvcc
CFLAGS2 = -arch sm_86 -g -G

all: serial cuda

cuda: mnist_nn_cuda.cu
	$(COMPILER2) mnist_nn_cuda.cu -o cuda $(CFLAGS2)

serial: mnist_nn_serial_c.c
	$(COMPILER1) mnist_nn_serial_c.c -o serial $(CFLAGS1)

clean:
	rm serial cuda

re: clean all
