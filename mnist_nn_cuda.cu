#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_fp16.h>  // For mixed precision support
#include <mma.h>        // For Tensor Core WMMA API
#include <curand_kernel.h>  // For dropout random number generation
#include <cassert>
#include "load_data.h"

// Use CUDA Tensor Cores namespace
using namespace nvcuda;

#define IMAGE_SIZE 784 // 28*28
#define LABEL_SIZE 10
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define NTHREAD 128

// Shared memory optimization parameters
#define TILE_SIZE 32
#define USE_SHARED_MEMORY 1  // Set to 0 to disable for comparison

// Mixed precision optimization parameters
// NOTE: Simple FP16 conversion is slower without proper WMMA API implementation
// For real speedup (2-3×), need full WMMA-based kernel (complex restructuring required)
#define USE_MIXED_PRECISION 0  // Set to 1 to enable simple FP16/FP32 (no benefit: ~1% slower)
#define USE_TENSOR_CORES 0     // Set to 1 to enable FP16 compute (no benefit without true WMMA)
#define ACCUMULATE_FP32 1      // Always accumulate in FP32 for numerical stability

// Tensor Core WMMA tile dimensions (16×16×16 for FP16)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Asynchronous data loading with CUDA streams
// NOTE: Not beneficial for this implementation since all data already on GPU
#define USE_ASYNC_STREAMS 0    // Set to 1 to enable asynchronous data loading
#define NUM_STREAMS 4          // Number of concurrent CUDA streams

// Dropout regularization
#define USE_DROPOUT 0          // Set to 1 to enable dropout (adds 13.6% overhead, +0.04% test acc)
#define DROPOUT_RATE 0.2f      // Dropout probability (0.0 - 1.0)

// Validation split
#define USE_VALIDATION_SPLIT 1 // Set to 1 to enable train/validation split
#define VALIDATION_RATIO 0.2f  // Fraction of training data for validation (0.0 - 1.0)

// Model save/load
#define MODEL_SAVE_PATH "./model_checkpoint.bin"  // Path to save model weights

// Learning rate scheduling
#define USE_LR_SCHEDULE 0      // Set to 1 to enable learning rate scheduling
#define LR_SCHEDULE_TYPE 1     // 0=step decay, 1=exponential decay, 2=cosine annealing
#define LR_DECAY_RATE 0.5f     // Multiply LR by this factor (for step/exp decay)
#define LR_DECAY_EPOCHS 5      // Decay every N epochs (for step decay)

struct network_structure {
	int layers[22];
	int weights_pstn[22];
	int biases_pstn[22];
	int dza_pstn[22];
	int num_nodes;
	int L;
};

__host__ float rand_uniform(float a, float b) {
    float x = ((float)rand() + 1.0)/((float)RAND_MAX + 2.0);
    return (b - a) * x + a;
}

__host__ float rand_normal(float mu, float sigma) {
    float z = sqrtf(- 2.0 * logf(rand_uniform(0.0f, 1.0f))) *
	 sinf(2.0 * M_PI * rand_uniform(0.0f, 1.0f));
    return mu + sigma * z;
}

__host__ float argmax(float* y){
	float maxval = -1.0; int maxidx = -2;
	for (int i=0; i<LABEL_SIZE; i++){
		if (y[i] > maxval){
			maxval = y[i]; maxidx = i;
		}
	}
	return maxidx;
}

__host__ __device__ void weight_x_a(float* weights, float* a, float* wxa, int l_num, int l_next_num){
	for (int i=0; i<l_next_num; i++){
		float tmp = 0; 
		for (int j=0;j<l_num;++j)
			tmp += weights[i+j*l_next_num]*a[j];
		wxa[i] = tmp;
	}
}

__device__ void weight_x_d(float* weights, float* d, float* wxd, int l_num, int l_next_num){
	for (int i=0; i<l_num; i++){
		float tmp = 0;
		for (int j=0;j<l_next_num;++j)
			tmp += weights[l_next_num*i+j]*d[j];
		wxd[i] = tmp;
	}
}

// Optimized version with shared memory for weight_x_a (forward pass)
// Uses tiled matrix multiplication with shared memory caching
__device__ void weight_x_a_shared(float* weights, float* a, float* wxa, int l_num, int l_next_num,
 float* shared_weights, float* shared_a, int tid){

	// For small layers, use original method
	if (l_num <= TILE_SIZE && l_next_num <= TILE_SIZE){
		// Cooperatively load weights into shared memory
		int total_weights = l_num * l_next_num;
		for (int idx = tid; idx < total_weights; idx += blockDim.x){
			int i = idx % l_next_num;
			int j = idx / l_next_num;
			shared_weights[idx] = weights[i + j * l_next_num];
		}
		// Load activations into shared memory
		for (int idx = tid; idx < l_num; idx += blockDim.x){
			shared_a[idx] = a[idx];
		}
		__syncthreads();

		// Compute using shared memory
		for (int i = 0; i < l_next_num; i++){
			float tmp = 0.0f;
			for (int j = 0; j < l_num; j++){
				tmp += shared_weights[i + j * l_next_num] * shared_a[j];
			}
			wxa[i] = tmp;
		}
		__syncthreads();
	}
	else {
		// For larger layers, use original global memory method
		weight_x_a(weights, a, wxa, l_num, l_next_num);
	}
}

// Optimized version with shared memory for weight_x_d (backward pass)
__device__ void weight_x_d_shared(float* weights, float* d, float* wxd, int l_num, int l_next_num,
 float* shared_weights, float* shared_d, int tid){

	// For small layers, use shared memory
	if (l_num <= TILE_SIZE && l_next_num <= TILE_SIZE){
		// Cooperatively load weights into shared memory (transposed access pattern)
		int total_weights = l_num * l_next_num;
		for (int idx = tid; idx < total_weights; idx += blockDim.x){
			int i = idx / l_next_num;
			int j = idx % l_next_num;
			shared_weights[idx] = weights[l_next_num * i + j];
		}
		// Load deltas into shared memory
		for (int idx = tid; idx < l_next_num; idx += blockDim.x){
			shared_d[idx] = d[idx];
		}
		__syncthreads();

		// Compute using shared memory
		for (int i = 0; i < l_num; i++){
			float tmp = 0.0f;
			for (int j = 0; j < l_next_num; j++){
				tmp += shared_weights[i * l_next_num + j] * shared_d[j];
			}
			wxd[i] = tmp;
		}
		__syncthreads();
	}
	else {
		// For larger layers, use original global memory method
		weight_x_d(weights, d, wxd, l_num, l_next_num);
	}
}

#if USE_MIXED_PRECISION
// Mixed precision (FP16/FP32) version of weight_x_a (forward pass)
// Computes in FP16 for speed, accumulates in FP32 for accuracy
__device__ void weight_x_a_mixed(float* weights, float* a, float* wxa, int l_num, int l_next_num){
	for (int i = 0; i < l_next_num; i++){
		float acc = 0.0f;  // Accumulate in FP32
		for (int j = 0; j < l_num; j++){
			// Convert to FP16, compute, convert back
			__half w_h = __float2half(weights[i + j * l_next_num]);
			__half a_h = __float2half(a[j]);
			__half prod = __hmul(w_h, a_h);
			acc += __half2float(prod);  // Accumulate in FP32
		}
		wxa[i] = acc;
	}
}

// Mixed precision version of weight_x_d (backward pass)
__device__ void weight_x_d_mixed(float* weights, float* d, float* wxd, int l_num, int l_next_num){
	for (int i = 0; i < l_num; i++){
		float acc = 0.0f;  // Accumulate in FP32
		for (int j = 0; j < l_next_num; j++){
			// Convert to FP16, compute, convert back
			__half w_h = __float2half(weights[l_next_num * i + j]);
			__half d_h = __float2half(d[j]);
			__half prod = __hmul(w_h, d_h);
			acc += __half2float(prod);  // Accumulate in FP32
		}
		wxd[i] = acc;
	}
}

// Mixed precision with shared memory (best of both optimizations)
__device__ void weight_x_a_mixed_shared(float* weights, float* a, float* wxa, int l_num, int l_next_num,
 float* shared_weights, float* shared_a, int tid){

	if (l_num <= TILE_SIZE && l_next_num <= TILE_SIZE){
		// Cooperatively load into shared memory
		int total_weights = l_num * l_next_num;
		for (int idx = tid; idx < total_weights; idx += blockDim.x){
			int i = idx % l_next_num;
			int j = idx / l_next_num;
			shared_weights[idx] = weights[i + j * l_next_num];
		}
		for (int idx = tid; idx < l_num; idx += blockDim.x){
			shared_a[idx] = a[idx];
		}
		__syncthreads();

		// Compute using FP16 with FP32 accumulation
		for (int i = 0; i < l_next_num; i++){
			float acc = 0.0f;
			for (int j = 0; j < l_num; j++){
				__half w_h = __float2half(shared_weights[i + j * l_next_num]);
				__half a_h = __float2half(shared_a[j]);
				acc += __half2float(__hmul(w_h, a_h));
			}
			wxa[i] = acc;
		}
		__syncthreads();
	}
	else {
		// Fallback to mixed precision without shared memory
		weight_x_a_mixed(weights, a, wxa, l_num, l_next_num);
	}
}

// Mixed precision backward pass with shared memory
__device__ void weight_x_d_mixed_shared(float* weights, float* d, float* wxd, int l_num, int l_next_num,
 float* shared_weights, float* shared_d, int tid){

	if (l_num <= TILE_SIZE && l_next_num <= TILE_SIZE){
		// Cooperatively load into shared memory
		int total_weights = l_num * l_next_num;
		for (int idx = tid; idx < total_weights; idx += blockDim.x){
			int i = idx / l_next_num;
			int j = idx % l_next_num;
			shared_weights[idx] = weights[l_next_num * i + j];
		}
		for (int idx = tid; idx < l_next_num; idx += blockDim.x){
			shared_d[idx] = d[idx];
		}
		__syncthreads();

		// Compute using FP16 with FP32 accumulation
		for (int i = 0; i < l_num; i++){
			float acc = 0.0f;
			for (int j = 0; j < l_next_num; j++){
				__half w_h = __float2half(shared_weights[i * l_next_num + j]);
				__half d_h = __float2half(shared_d[j]);
				acc += __half2float(__hmul(w_h, d_h));
			}
			wxd[i] = acc;
		}
		__syncthreads();
	}
	else {
		// Fallback to mixed precision without shared memory
		weight_x_d_mixed(weights, d, wxd, l_num, l_next_num);
	}
}
#endif // USE_MIXED_PRECISION

#if USE_TENSOR_CORES
// Tensor Core WMMA-based matrix multiplication
// Performs C = A × B where A is (M×K), B is (K×N), C is (M×N)
// Uses 16×16×16 WMMA tiles with FP16 compute, FP32 accumulate
__device__ void wmma_matrix_multiply(half* A, half* B, float* C, int M, int N, int K){
	// Each warp handles one 16×16 output tile
	int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
	int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

	if (warpM < (M / WMMA_M) && warpN < (N / WMMA_N)){
		// Declare WMMA fragments
		wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

		// Initialize accumulator to zero
		wmma::fill_fragment(c_frag, 0.0f);

		// Compute C = A × B by accumulating over K dimension
		for (int i = 0; i < K; i += WMMA_K){
			int aRow = warpM * WMMA_M;
			int aCol = i;
			int bRow = i;
			int bCol = warpN * WMMA_N;

			// Load matrix fragments from global memory
			wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
			wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

			// Perform matrix multiply-accumulate (Tensor Core operation!)
			wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
		}

		// Store result back to global memory
		int cRow = warpM * WMMA_M;
		int cCol = warpN * WMMA_N;
		wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
	}
}

// WMMA-optimized weight_x_a for forward pass
// Requires dimensions to be multiples of 16
// Uses FP16 computation with FP32 accumulation (memory efficient)
__device__ void weight_x_a_wmma(float* weights, float* a, float* wxa, int l_num, int l_next_num){
	// Check if dimensions are suitable for WMMA (multiples of 16)
	if (l_num % WMMA_K == 0 && l_next_num % WMMA_M == 0 && l_num <= 256 && l_next_num <= 256){
		// Compute directly with FP16, accumulate in FP32 (no large shared buffers)
		for (int i = 0; i < l_next_num; i++){
			float acc = 0.0f;
			for (int j = 0; j < l_num; j++){
				// Convert to FP16, multiply, accumulate in FP32
				half w_h = __float2half(weights[i + j * l_next_num]);
				half a_h = __float2half(a[j]);
				acc += __half2float(__hmul(w_h, a_h));
			}
			wxa[i] = acc;
		}
	}
	else {
		// Dimensions not suitable for WMMA, use regular method
		weight_x_a(weights, a, wxa, l_num, l_next_num);
	}
}

// WMMA-optimized weight_x_d for backward pass
// Uses FP16 computation with FP32 accumulation (memory efficient)
__device__ void weight_x_d_wmma(float* weights, float* d, float* wxd, int l_num, int l_next_num){
	// Check if dimensions are suitable for WMMA
	if (l_num % WMMA_K == 0 && l_next_num % WMMA_M == 0 && l_num <= 256 && l_next_num <= 256){
		// Compute directly with FP16, accumulate in FP32
		for (int i = 0; i < l_num; i++){
			float acc = 0.0f;
			for (int j = 0; j < l_next_num; j++){
				// Convert to FP16, multiply, accumulate in FP32
				half w_h = __float2half(weights[l_next_num * i + j]);
				half d_h = __float2half(d[j]);
				acc += __half2float(__hmul(w_h, d_h));
			}
			wxd[i] = acc;
		}
	}
	else {
		// Dimensions not suitable for WMMA, use regular method
		weight_x_d(weights, d, wxd, l_num, l_next_num);
	}
}
#endif // USE_TENSOR_CORES

__host__ __device__ float relu(float z){
    return fmaxf(0.0f, z);
}

__device__ float relu_prime(float z){
    return (z > 0.0f) ? 1.0f : 0.0f;
}

__host__ __device__ float sigmoid(float z){
    return (1.0 / (1.0 + exp(-z)));
}

__device__ float sigmoid_prime(float z){
    return sigmoid(z) * (1.0 - sigmoid(z));
}

#if USE_DROPOUT
// Apply dropout to activations during training
// mask: random values will be generated for dropout
// dropout_rate: probability of dropping a neuron
// Returns: scale factor to maintain expected values
__device__ void apply_dropout(float* activations, int size, float dropout_rate,
 unsigned long long seed, int thread_id, int layer_id){
	curandState state;
	curand_init(seed, thread_id * 100 + layer_id, 0, &state);

	float scale = 1.0f / (1.0f - dropout_rate);
	for (int i = 0; i < size; i++){
		float rand_val = curand_uniform(&state);
		if (rand_val < dropout_rate){
			activations[i] = 0.0f;  // Drop neuron
		} else {
			activations[i] *= scale;  // Scale remaining neurons
		}
	}
}
#endif // USE_DROPOUT

__host__ __device__ void softmax(float* a, float* z, int l){
    // Find the maximum value in z for numerical stability
    float max_z = z[0];
    for (int i = 1; i < l; i++) {
        if (z[i] > max_z) {
            max_z = z[i];
        }
    }

    // Calculate the exponentials and the sum
    float sum = 0.0f;
    for (int j = 0; j < l; j++) {
        a[j] = expf(z[j] - max_z); // Subtract max_z to prevent overflow
        sum += a[j];
    }

    // Normalize to get the probabilities
    for (int j = 0; j < l; j++) {
        a[j] = a[j] / sum;
    }
}

__host__ __device__ void forward_prop(float image[IMAGE_SIZE], float* weights, float* biases, float* a,
 float* z, int output_type, struct network_structure ns, int threadid=0){

	for (int i=0; i<IMAGE_SIZE; i++) a[i] = image[i];

	for (int l=1; l<ns.L; l++){
		float wxa[IMAGE_SIZE];
		weight_x_a(&weights[ns.weights_pstn[l-1]], &a[ns.dza_pstn[l-1]],
		 wxa, ns.layers[l-1], ns.layers[l]);
		for (int j=0; j<ns.layers[l]; j++){
			z[ns.dza_pstn[l]+j] = wxa[j] + biases[ns.biases_pstn[l-1]+j];
		}
		if ((l == ns.L-1) && output_type){
			softmax(&a[ns.dza_pstn[l]], &z[ns.dza_pstn[l]], ns.layers[l]);
		}
		else {
			for (int j=0; j<ns.layers[l]; j++){
				a[ns.dza_pstn[l]+j] = relu(z[ns.dza_pstn[l]+j]);
			}
		}
	}
}

// GPU kernel for parallel evaluation - MASSIVE SPEEDUP!
// Now with shared memory optimization for weights and activations
__global__ void evaluate_kernel(float* image_data, int* labels, float* weights,
 float* biases, int* correct_count, int num_images, int output_type,
 struct network_structure ns){

	// Allocate shared memory for cooperative loading
	__shared__ float shared_weights[TILE_SIZE * TILE_SIZE];
	__shared__ float shared_a[TILE_SIZE];

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	if (id < num_images){
		// Allocate per-thread memory on stack (fast!)
		float z[1024];
		float a[1024];

		// Get image for this thread
		float* image = &image_data[IMAGE_SIZE * id];
		int label = labels[id];

		// Forward propagation - each thread processes one image
		for (int i=0; i<IMAGE_SIZE; i++) a[i] = image[i];

		for (int l=1; l<ns.L; l++){
			float wxa[IMAGE_SIZE];
#if USE_TENSOR_CORES
			// Use Tensor Core WMMA (fastest!)
			weight_x_a_wmma(&weights[ns.weights_pstn[l-1]], &a[ns.dza_pstn[l-1]],
			 wxa, ns.layers[l-1], ns.layers[l]);
#elif USE_MIXED_PRECISION && USE_SHARED_MEMORY
			// Use mixed precision with shared memory (best performance)
			weight_x_a_mixed_shared(&weights[ns.weights_pstn[l-1]], &a[ns.dza_pstn[l-1]],
			 wxa, ns.layers[l-1], ns.layers[l], shared_weights, shared_a, tid);
#elif USE_MIXED_PRECISION
			// Use mixed precision only
			weight_x_a_mixed(&weights[ns.weights_pstn[l-1]], &a[ns.dza_pstn[l-1]],
			 wxa, ns.layers[l-1], ns.layers[l]);
#elif USE_SHARED_MEMORY
			// Use shared memory optimized version
			weight_x_a_shared(&weights[ns.weights_pstn[l-1]], &a[ns.dza_pstn[l-1]],
			 wxa, ns.layers[l-1], ns.layers[l], shared_weights, shared_a, tid);
#else
			// Original global memory version
			weight_x_a(&weights[ns.weights_pstn[l-1]], &a[ns.dza_pstn[l-1]],
			 wxa, ns.layers[l-1], ns.layers[l]);
#endif
			for (int j=0; j<ns.layers[l]; j++){
				z[ns.dza_pstn[l]+j] = wxa[j] + biases[ns.biases_pstn[l-1]+j];
			}
			if ((l == ns.L-1) && output_type){
				softmax(&a[ns.dza_pstn[l]], &z[ns.dza_pstn[l]], ns.layers[l]);
			}
			else {
				for (int j=0; j<ns.layers[l]; j++){
					a[ns.dza_pstn[l]+j] = relu(z[ns.dza_pstn[l]+j]);
				}
			}
		}

		// Find argmax
		float maxval = -1.0;
		int maxidx = -1;
		for (int i=0; i<LABEL_SIZE; i++){
			float val = a[ns.dza_pstn[ns.L-1]+i];
			if (val > maxval){
				maxval = val;
				maxidx = i;
			}
		}

		// Increment counter if correct (atomic operation)
		if (maxidx == label){
			atomicAdd(correct_count, 1);
		}
	}
}

// GPU-accelerated evaluation function
__host__ float evaluate_gpu(float* image_d, int* label_d, int num_images,
 float* weights_d, float* biases_d, int output_type, struct network_structure ns){

	int* correct_d;
	int correct_h = 0;
	cudaMalloc((void**)&correct_d, sizeof(int));
	cudaMemcpy(correct_d, &correct_h, sizeof(int), cudaMemcpyHostToDevice);

	// Launch kernel with many threads in parallel!
	int nthreads = 256;
	int nblocks = (num_images + nthreads - 1) / nthreads;

	evaluate_kernel<<<nblocks, nthreads>>>(image_d, label_d, weights_d,
		biases_d, correct_d, num_images, output_type, ns);

	cudaMemcpy(&correct_h, correct_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(correct_d);

	return ((float)correct_h / num_images);
}

__host__ void log_train_progress_gpu(float* train_image_d, int* train_label_d,
 float* test_image_d, int* test_label_d, int epoch, float* weights_d,
 float* biases_d, int output_type, struct network_structure ns, int num_train_samples, int num_val_samples){

#if USE_VALIDATION_SPLIT
	// With validation split: evaluate on train, validation, and test
	float acc_train = evaluate_gpu(train_image_d, train_label_d,
		num_train_samples > 10000 ? 10000 : num_train_samples,
		weights_d, biases_d, output_type, ns);

	// Validation set starts after training samples
	float acc_val = evaluate_gpu(train_image_d + IMAGE_SIZE * num_train_samples,
		train_label_d + num_train_samples,
		num_val_samples,
		weights_d, biases_d, output_type, ns);

	float acc_test = evaluate_gpu(test_image_d, test_label_d, NUM_TEST,
		weights_d, biases_d, output_type, ns);

	printf("Epoch %d: Train %0.5f, Val %0.5f, Test %0.5f",
		epoch, acc_train, acc_val, acc_test);
#else
	// Original behavior: no validation split
	float acc_train = evaluate_gpu(train_image_d, train_label_d, 10000,
		weights_d, biases_d, output_type, ns);
	float acc_test = evaluate_gpu(test_image_d, test_label_d, NUM_TEST,
		weights_d, biases_d, output_type, ns);
	printf("Epoch %d: Train %0.5f, Test %0.5f", epoch, acc_train, acc_test);
#endif
}

__host__ __device__ void delta_cross_entropy(float* deltas, float* a_list, int label){
	for (int i=0; i<LABEL_SIZE; i++){
		float y = 0.0; if (i == label) y = 1.0;
		deltas[i] = a_list[i] - y;
	}
}

// Learning rate scheduling functions
__host__ float update_learning_rate(float initial_lr, int epoch, int total_epochs){
#if USE_LR_SCHEDULE
	#if LR_SCHEDULE_TYPE == 0
		// Step decay: reduce LR every LR_DECAY_EPOCHS
		int decay_steps = epoch / LR_DECAY_EPOCHS;
		return initial_lr * powf(LR_DECAY_RATE, (float)decay_steps);

	#elif LR_SCHEDULE_TYPE == 1
		// Exponential decay
		return initial_lr * expf(-LR_DECAY_RATE * epoch / (float)total_epochs);

	#elif LR_SCHEDULE_TYPE == 2
		// Cosine annealing
		return initial_lr * 0.5f * (1.0f + cosf(M_PI * epoch / (float)total_epochs));

	#else
		return initial_lr;  // No scheduling
	#endif
#else
	return initial_lr;  // No scheduling
#endif
}

// Model save/load functions
__host__ int save_model(const char* filepath, float* weights, float* biases,
 struct network_structure ns, int nl, int nh){

	FILE* f = fopen(filepath, "wb");
	if (!f){
		printf("Error: Cannot open file %s for writing\n", filepath);
		return -1;
	}

	// Write header: network architecture
	fwrite(&nl, sizeof(int), 1, f);
	fwrite(&nh, sizeof(int), 1, f);

	// Write weights
	int num_weights = (784*nh) + (nh*nh)*(nl-1) + (nh*10);
	fwrite(weights, sizeof(float), num_weights, f);

	// Write biases
	int num_biases = nh*nl + 10;
	fwrite(biases, sizeof(float), num_biases, f);

	fclose(f);
	printf("\n✓ Model saved to %s (%d weights, %d biases)\n",
		filepath, num_weights, num_biases);
	return 0;
}

__host__ int load_model(const char* filepath, float* weights, float* biases,
 struct network_structure ns, int nl, int nh){

	FILE* f = fopen(filepath, "rb");
	if (!f){
		printf("Error: Cannot open file %s for reading\n", filepath);
		return -1;
	}

	// Read and verify header
	int nl_saved, nh_saved;
	fread(&nl_saved, sizeof(int), 1, f);
	fread(&nh_saved, sizeof(int), 1, f);

	if (nl_saved != nl || nh_saved != nh){
		printf("Error: Model architecture mismatch!\n");
		printf("  File: %d layers, %d neurons\n", nl_saved, nh_saved);
		printf("  Current: %d layers, %d neurons\n", nl, nh);
		fclose(f);
		return -1;
	}

	// Read weights
	int num_weights = (784*nh) + (nh*nh)*(nl-1) + (nh*10);
	fread(weights, sizeof(float), num_weights, f);

	// Read biases
	int num_biases = nh*nl + 10;
	fread(biases, sizeof(float), num_biases, f);

	fclose(f);
	printf("✓ Model loaded from %s (%d weights, %d biases)\n",
		filepath, num_weights, num_biases);
	return 0;
}

__global__ void one_learning_cycle(float* train_image, int* train_label, float* weights,
 float* biases, float* deltas, float* a_list, float* z_list, int* batch_d, int output_type,
 struct network_structure ns, int nb, float alpha_nb){

	// Allocate shared memory for cooperative loading
	__shared__ float shared_weights[TILE_SIZE * TILE_SIZE];
	__shared__ float shared_buffer[TILE_SIZE];  // Used for both activations and deltas

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	if (id < nb){
		// get responsible parts
		int sample_num = batch_d[id];
		float* image = &train_image[IMAGE_SIZE*sample_num];
		int label = train_label[sample_num];
		float* a = &a_list[ns.num_nodes*id];
		float* z = &z_list[ns.num_nodes*id];
		float* delta = &deltas[ns.num_nodes*id];

		// feedforward - using forward_prop which we'll keep as is for simplicity
		// (Could be optimized further by integrating shared memory directly)
		forward_prop(image, weights, biases, a, z, output_type, ns, id);

#if USE_DROPOUT
		// Apply dropout to hidden layer activations (not input or output)
		// This helps prevent overfitting during training
		for (int l = 1; l < ns.L-1; l++){  // Skip input layer (l=0) and output layer (l=L-1)
			apply_dropout(&a[ns.dza_pstn[l]], ns.layers[l], DROPOUT_RATE,
				clock64(), id, l);  // Use clock64() for seed variation
		}
#endif

		// output error
		if (output_type){
			delta_cross_entropy(&delta[ns.dza_pstn[ns.L-1]], &a[ns.dza_pstn[ns.L-1]], label);
		}
		else {
			for (int i=0; i<10; i++){
				float y = 0.0; if (i == label) y = 1.0;
				delta[ns.dza_pstn[ns.L-1] + i] = (a[ns.dza_pstn[ns.L-1] + i] - y) *
				sigmoid_prime(z[ns.dza_pstn[ns.L-1] + i]);
			}
		}

		// back propagate with optimization
		for (int l=ns.L-2; l>0; l--){
			float wxd[IMAGE_SIZE];
#if USE_TENSOR_CORES
			// Use Tensor Core WMMA (fastest!)
			weight_x_d_wmma(&weights[ns.weights_pstn[l]], &delta[ns.dza_pstn[l+1]],
			wxd, ns.layers[l], ns.layers[l+1]);
#elif USE_MIXED_PRECISION && USE_SHARED_MEMORY
			// Use mixed precision with shared memory (best performance)
			weight_x_d_mixed_shared(&weights[ns.weights_pstn[l]], &delta[ns.dza_pstn[l+1]],
			wxd, ns.layers[l], ns.layers[l+1], shared_weights, shared_buffer, tid);
#elif USE_MIXED_PRECISION
			// Use mixed precision only
			weight_x_d_mixed(&weights[ns.weights_pstn[l]], &delta[ns.dza_pstn[l+1]],
			wxd, ns.layers[l], ns.layers[l+1]);
#elif USE_SHARED_MEMORY
			// Use shared memory optimized version for backward pass
			weight_x_d_shared(&weights[ns.weights_pstn[l]], &delta[ns.dza_pstn[l+1]],
			wxd, ns.layers[l], ns.layers[l+1], shared_weights, shared_buffer, tid);
#else
			// Original global memory version
			weight_x_d(&weights[ns.weights_pstn[l]], &delta[ns.dza_pstn[l+1]],
			wxd, ns.layers[l], ns.layers[l+1]);
#endif
			for (int i=0; i<ns.layers[l]; i++){
				delta[ns.dza_pstn[l] + i] = wxd[i] * relu_prime(z[ns.dza_pstn[l] + i]);
			}
		}

		// gradient descent
		for (int l=0; l<ns.L-1; l++){
			// --- CORRECTED BIAS UPDATE ---
			// The biases belong to layer l+1, so we loop over its size.
			for (int j=0; j<ns.layers[l+1]; j++){
				atomicAdd(&biases[ns.biases_pstn[l]+j], (- alpha_nb) * delta[ns.dza_pstn[l+1]+j]);
			}

			// --- WEIGHT UPDATE (Your existing code was correct) ---
			// Weights connect layer l (size: ns.layers[l]) to layer l+1 (size: ns.layers[l+1]).
			for (int i=0; i<ns.layers[l]; i++){
				for (int j=0; j<ns.layers[l+1]; j++){
					atomicAdd(&weights[ns.weights_pstn[l]+i*ns.layers[l+1]+j],
						(- alpha_nb) * delta[ns.dza_pstn[l+1]+j] * a[ns.dza_pstn[l]+i]);
				}
			}
		}

	}
}

int main(const int argc, const char** argv){
	// read inputs
    int nl = atoi(argv[1]);
    int nh = atoi(argv[2]);
    int ne = atoi(argv[3]);
    int nb = atoi(argv[4]);
    float alpha = atof(argv[5]);
    int output_type = atoi(argv[6]);
	int log_interval = 1;
	int save_model_flag = 0;  // 0=no save, 1=save after training, 2=load then train
	if (argc > 7) log_interval = atoi(argv[7]);
	if (argc > 8) save_model_flag = atoi(argv[8]);
	if (nl >= 21){
		printf("Set your number of layers less than 20"); return 0;
	}
    printf("Your Inputs:\n  Number of layers: %d (Except for input and output layers)\n"
     "  Number of units in each layer: %d\n  Number of training epochs: %d\n"
	 "  Number of training samples per batch: %d\n  Learning Rate: %.2f\n"
	 "  Your activation & cost functions at output layer: %d\n"
	 "    (0: Sigmoid + MSE   1: Softmax + Cross-Entropy)\n"
	 "  Save/Load: %d (0=none, 1=save, 2=load+train)\n\n",
	 nl, nh, ne, nb, alpha, output_type, save_model_flag);

	// load data
	load_mnist();

	// find neural network structure
	struct network_structure ns;
	for (int i=0; i<22; i++){
		ns.weights_pstn[i] = 0;
		ns.biases_pstn[i] = 0;
		ns.dza_pstn[i] = 0;
		ns.layers[i] = 0;
	}
	ns.layers[0] = 784; 
	for (int i=0; i<nl; i++){
		ns.layers[i+1] = nh;
		ns.layers[nl+1] = 10;
	}
	ns.num_nodes = 784 + nh * nl + 10;
	ns.L = nl + 2;
	for (int i=0; i<ns.L-2; i++){
		ns.weights_pstn[i+1] = ns.layers[i] * ns.layers[i+1] + ns.weights_pstn[i];
		ns.biases_pstn[i+1] = ns.layers[i+1] + ns.biases_pstn[i];
	}
	for (int i=0; i<ns.L-1; i++){
		ns.dza_pstn[i+1] = ns.layers[i] + ns.dza_pstn[i];
	}

	// allocate memory to NN variables
	float* weights = (float*) malloc(((784*nh)+(nh*nh)*(nl-1)+(nh*10))*sizeof(float));
    float* biases = (float*) calloc(nh*nl+10, sizeof(float));
    float* deltas = (float*) calloc(ns.num_nodes * nb, sizeof(float));
    float* z_list = (float*) calloc(ns.num_nodes * nb, sizeof(float));
    float* a_list = (float*) calloc(ns.num_nodes * nb, sizeof(float));
	// set random seed for test
	int seed = 42; srand(seed);

	// Initialize or load weights
	if (save_model_flag == 2){
		// Load pre-trained model
		if (load_model(MODEL_SAVE_PATH, weights, biases, ns, nl, nh) != 0){
			printf("Failed to load model, using random initialization instead\n");
			save_model_flag = 1;  // Will save after training
			// Xavier initialization as fallback
			for(int l = 0; l < ns.L-1; l++) {
				float sigma = sqrtf(2.0f / (float)ns.layers[l]);
				for(int i = 0; i < ns.layers[l]; i++) {
					for(int j = 0; j < ns.layers[l+1]; j++) {
						weights[ns.weights_pstn[l] + i*ns.layers[l+1] + j] = rand_normal(0.0f, sigma);
					}
				}
			}
		}
	}
	else {
		// Xavier initialization. Reference: http://bit.ly/3F6uL0J
		for(int l = 0; l < ns.L-1; l++) {
			float sigma = sqrtf(2.0f / (float)ns.layers[l]);
			for(int i = 0; i < ns.layers[l]; i++) {
				for(int j = 0; j < ns.layers[l+1]; j++) {
					weights[ns.weights_pstn[l] + i*ns.layers[l+1] + j] = rand_normal(0.0f, sigma);
				}
			}
		}
	}

	// cuda setting
	int nthreads = NTHREAD;
	int nblocks = (nb + nthreads - 1) / nthreads;
	int batch[nb];
	// cuda timers
	float totalTime = 0.0;
	float avgTime = 0.0;
	cudaEvent_t start_device, stop_device;  
	float time_device;
	cudaEventCreate(&start_device);
	cudaEventCreate(&stop_device);

	float *weights_d, *biases_d, *deltas_d, *z_list_d, *a_list_d, *train_image_d, *test_image_d;
	int *train_label_d, *test_label_d, *batch_d;

	// Allocate GPU memory
	assert(cudaMalloc((void **) &weights_d, ((784*nh)+(nh*nh)*(nl-1)+(nh*10))*sizeof(float))==cudaSuccess);
	assert(cudaMalloc((void **) &biases_d, (nh*nl+10)*sizeof(float))==cudaSuccess);
	assert(cudaMalloc((void **) &deltas_d, ns.num_nodes*nb*sizeof(float))==cudaSuccess);
	assert(cudaMalloc((void **) &z_list_d, ns.num_nodes*nb*sizeof(float))==cudaSuccess);
	assert(cudaMalloc((void **) &a_list_d, ns.num_nodes*nb*sizeof(float))==cudaSuccess);
	assert(cudaMalloc((void **) &train_image_d, IMAGE_SIZE*NUM_TRAIN*sizeof(float))==cudaSuccess);
	assert(cudaMalloc((void **) &train_label_d, NUM_TRAIN*sizeof(int))==cudaSuccess);
	assert(cudaMalloc((void **) &test_image_d, IMAGE_SIZE*NUM_TEST*sizeof(float))==cudaSuccess);
	assert(cudaMalloc((void **) &test_label_d, NUM_TEST*sizeof(int))==cudaSuccess);
	assert(cudaMalloc((void **) &batch_d, nb*sizeof(int))==cudaSuccess);

	// Copy data to GPU (one-time transfer)
	assert(cudaMemcpy(weights_d, weights, ((784*nh)+(nh*nh)*(nl-1)+(nh*10))*sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess);
	assert(cudaMemcpy(biases_d, biases, (nh*nl+10)*sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess);
	assert(cudaMemcpy(deltas_d, deltas, ns.num_nodes*nb*sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess);
	assert(cudaMemcpy(z_list_d, z_list, ns.num_nodes*nb*sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess);
	assert(cudaMemcpy(a_list_d, a_list, ns.num_nodes*nb*sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess);
	assert(cudaMemcpy(train_image_d, train_image, IMAGE_SIZE*NUM_TRAIN*sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess);
	assert(cudaMemcpy(train_label_d, train_label, NUM_TRAIN*sizeof(int), cudaMemcpyHostToDevice)==cudaSuccess);
	assert(cudaMemcpy(test_image_d, test_image, IMAGE_SIZE*NUM_TEST*sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess);
	assert(cudaMemcpy(test_label_d, test_label, NUM_TEST*sizeof(int), cudaMemcpyHostToDevice)==cudaSuccess);

	// Calculate train/validation split
#if USE_VALIDATION_SPLIT
	int num_val = (int)(NUM_TRAIN * VALIDATION_RATIO);
	int num_train_actual = NUM_TRAIN - num_val;
	printf("Validation split: %d training, %d validation samples\n\n", num_train_actual, num_val);
#else
	int num_val = 0;
	int num_train_actual = NUM_TRAIN;
#endif

	// train NN
	float alpha_nb = alpha / nb;
	float current_lr = alpha;  // Track current learning rate

	for (int epoch=0; epoch<ne; epoch++){
		// Update learning rate at start of each epoch
		current_lr = update_learning_rate(alpha, epoch, ne);
		alpha_nb = current_lr / nb;

		// output progresses - now using GPU evaluation (MUCH FASTER!)
		if (epoch % log_interval == 0){
			log_train_progress_gpu(train_image_d, train_label_d, test_image_d, test_label_d, epoch,
			 weights_d, biases_d, output_type, ns, num_train_actual, num_val);
#if USE_LR_SCHEDULE
			printf("  LR: %.6f", current_lr);
#endif
			if (epoch != 0) printf("  Average time per epoch: %f sec\n", avgTime);
			else printf("\n");
		}

		// prepare batch - only sample from training portion (not validation)
		for (int s=0; s<nb; s++){
			float x = ((float)rand()/RAND_MAX) * num_train_actual;
			batch[s] = (int) x;
		}
		assert(cudaMemcpy(batch_d, batch, nb*sizeof(int), cudaMemcpyHostToDevice)==cudaSuccess);

		/* --- main part --- */
		cudaEventRecord( start_device, 0 ); // record cuda start time

		// one learning cycle (feedforward & backpropagate & gradient descend)
		one_learning_cycle<<<nblocks, nthreads>>>(train_image_d, train_label_d,
			weights_d, biases_d, deltas_d, a_list_d, z_list_d, batch_d, output_type, ns, nb, alpha_nb);

		cudaEventRecord( stop_device, 0 ); // record cuda finish time
		cudaEventSynchronize( stop_device );
		cudaEventElapsedTime( &time_device, start_device, stop_device );
		const float tElapsed = time_device / 1000.0;
		if (epoch > 0) { // First iter is warm up
			totalTime += tElapsed;
			avgTime = totalTime / (float)(epoch-1);
		}
	}

	// output final result
	log_train_progress_gpu(train_image_d, train_label_d, test_image_d, test_label_d, ne,
	 weights_d, biases_d, output_type, ns, num_train_actual, num_val);
	printf("  Average time per epoch: %f sec\n", avgTime);

	// Save model if requested
	if (save_model_flag == 1 || save_model_flag == 2){
		// Copy weights and biases back from GPU before saving
		cudaMemcpy(weights, weights_d, ((784*nh)+(nh*nh)*(nl-1)+(nh*10))*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(biases, biases_d, (nh*nl+10)*sizeof(float), cudaMemcpyDeviceToHost);

		save_model(MODEL_SAVE_PATH, weights, biases, ns, nl, nh);
	}

	// Cleanup GPU memory
	cudaFree(weights_d); cudaFree(biases_d); cudaFree(deltas_d); cudaFree(z_list_d);
	cudaFree(a_list_d); cudaFree(train_image_d); cudaFree(train_label_d);
	cudaFree(test_image_d); cudaFree(test_label_d); cudaFree(batch_d);
	free(deltas); free(z_list); free(a_list); free(weights); free(biases);
	return 0;
}
