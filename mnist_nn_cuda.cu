#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cassert>
#include "load_data.h"

#define IMAGE_SIZE 784 // 28*28
#define LABEL_SIZE 10
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define NTHREAD 128

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
__global__ void evaluate_kernel(float* image_data, int* labels, float* weights,
 float* biases, int* correct_count, int num_images, int output_type,
 struct network_structure ns){

	int id = threadIdx.x + blockDim.x * blockIdx.x;
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
 float* biases_d, int output_type, struct network_structure ns){

	// Evaluate entirely on GPU - Match serial version (10k train samples for speed)
	float acc_train = evaluate_gpu(train_image_d, train_label_d, 10000, weights_d, biases_d, output_type, ns);
	float acc_test = evaluate_gpu(test_image_d, test_label_d, NUM_TEST, weights_d, biases_d, output_type, ns);
	printf("Epoch %d: Train %0.5f, Test %0.5f", epoch, acc_train, acc_test);
}

__host__ __device__ void delta_cross_entropy(float* deltas, float* a_list, int label){
	for (int i=0; i<LABEL_SIZE; i++){
		float y = 0.0; if (i == label) y = 1.0;
		deltas[i] = a_list[i] - y;
	}
}

__global__ void one_learning_cycle(float* train_image, int* train_label, float* weights,
 float* biases, float* deltas, float* a_list, float* z_list, int* batch_d, int output_type,
 struct network_structure ns, int nb, float alpha_nb){

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < nb){
		// get responsible parts
		int sample_num = batch_d[id];
		float* image = &train_image[IMAGE_SIZE*sample_num];
		int label = train_label[sample_num];
		float* a = &a_list[ns.num_nodes*id];
		float* z = &z_list[ns.num_nodes*id];
		float* delta = &deltas[ns.num_nodes*id];

		// feedforward
		forward_prop(image, weights, biases, a, z, output_type, ns, id);

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

		// back propagate
		for (int l=ns.L-2; l>0; l--){
			float wxd[IMAGE_SIZE];
			weight_x_d(&weights[ns.weights_pstn[l]], &delta[ns.dza_pstn[l+1]],
			wxd, ns.layers[l], ns.layers[l+1]);
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
	if (argc > 7) log_interval = atoi(argv[7]);
	if (nl >= 21){
		printf("Set your number of layers less than 20"); return 0;
	}
    printf("Your Inputs:\n  Number of layers: %d (Except for input and output layers)\n"
     "  Number of units in each layer: %d\n  Number of training epochs: %d\n"
	 "  Number of training samples per batch: %d\n  Learning Rate: %.2f\n"
	 "  Your activation & cost functions at output layer: %d\n"
	 "    (0: Sigmoid + MSE   1: Softmax + Cross-Entropy)\n\n",
	 nl, nh, ne, nb, alpha, output_type);

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
	// Xavier initialization. Reference: http://bit.ly/3F6uL0J
    for(int l = 0; l < ns.L-1; l++) {
        float sigma = sqrtf(2.0f / (float)ns.layers[l]);
        for(int i = 0; i < ns.layers[l]; i++) {
            for(int j = 0; j < ns.layers[l+1]; j++) {
                weights[ns.weights_pstn[l] + i*ns.layers[l+1] + j] = rand_normal(0.0f, sigma);
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

	// train NN
	float alpha_nb = alpha / nb;
	for (int epoch=0; epoch<ne; epoch++){
		// output progresses - now using GPU evaluation (MUCH FASTER!)
		if (epoch % log_interval == 0){
			log_train_progress_gpu(train_image_d, train_label_d, test_image_d, test_label_d, epoch,
			 weights_d, biases_d, output_type, ns);
			if (epoch != 0) printf("  Average time per epoch: %f sec\n", avgTime);
			else printf("\n");
		}

		// prepare batch
		for (int s=0; s<nb; s++){
			float x = ((float)rand()/RAND_MAX) * NUM_TRAIN;
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
	 weights_d, biases_d, output_type, ns);
	printf("  Average time per epoch: %f sec\n", avgTime);

	// Cleanup GPU memory
	cudaFree(weights_d); cudaFree(biases_d); cudaFree(deltas_d); cudaFree(z_list_d);
	cudaFree(a_list_d); cudaFree(train_image_d); cudaFree(train_label_d);
	cudaFree(test_image_d); cudaFree(test_label_d); cudaFree(batch_d);
	free(deltas); free(z_list); free(a_list); free(weights); free(biases);
	return 0;
}
