/* Input Example:
 * ./serial 2 30 300 6000 0.1 1 100
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "load_data.h"

#define IMAGE_SIZE 784 // 28*28
#define LABEL_SIZE 10
#define NUM_TRAIN 60000
#define NUM_TEST 10000

struct network_structure {
	int layers[22];
	int weights_pstn[22];
	int biases_pstn[22];
	int dza_pstn[22];
	int num_nodes;
	int L;
};

float rand_uniform(float a, float b) {
    float x = ((float)rand() + 1.0)/((float)RAND_MAX + 2.0);
    return (b - a) * x + a;
}

float rand_normal(float mu, float sigma) {
    float z = sqrtf(- 2.0 * logf(rand_uniform(0.0f, 1.0f))) *
	 sinf(2.0 * M_PI * rand_uniform(0.0f, 1.0f));
    return mu + sigma * z;
}

float argmax(float* y){
	float maxval = -1.0; int maxidx = -2;
	for (int i=0; i<LABEL_SIZE; i++){
		if (y[i] > maxval){
			maxval = y[i]; maxidx = i;
		}
	}
	return maxidx;
}

void weight_x_a(float* weights, float* a, float* wxa, int l_num, int l_next_num){
	for (int i=0; i<l_next_num; i++){
		float tmp = 0; 
		for (int j=0;j<l_num;++j)
			tmp += weights[i+j*l_next_num]*a[j];
		wxa[i] = tmp;
	}
}

void weight_x_d(float* weights, float* d, float* wxd, int l_num, int l_next_num){
	for (int i=0; i<l_num; i++){
		float tmp = 0; 
		for (int j=0;j<l_next_num;++j)
			tmp += weights[l_next_num*i+j]*d[j];
		wxd[i] = tmp;
	}
}

float sigmoid(float z){
	return (1.0 / (1.0 + exp(-z)));
}

float sigmoid_prime(float z){
	return sigmoid(z) * (1.0 - sigmoid(z));
}

void softmax(float* a, float* z, int l){
	float sum = 0.0;
	for (int j=0; j<l; j++){
		a[j] = exp(z[j]);
		sum += exp(z[j]);
	}
	for (int j=0; j<l; j++){
		a[j] = a[j] / sum;
	}
}

void forward_prop(float image[IMAGE_SIZE], float* weights, float* biases, float* a,
 float* z, int output_type, struct network_structure ns){

	for (int i=0; i<IMAGE_SIZE; i++) a[i] = image[i];

	for (int l=1; l<ns.L; l++){
		float wxa[ns.layers[l]];
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
				a[ns.dza_pstn[l]+j] = sigmoid(z[ns.dza_pstn[l]+j]);
			}
		}
	}
}

float evaluate(float image[][IMAGE_SIZE], int num_images, int label[num_images],
 float* weights, float* biases, int output_type, struct network_structure ns){

	int ctr = 0;
	float yhat[LABEL_SIZE];
	for (int i=0; i<num_images; i++){
		float* z = (float*) calloc(ns.num_nodes, sizeof(float));
		float* a = (float*) calloc(ns.num_nodes, sizeof(float));
		forward_prop(image[i], weights, biases, a, z, output_type, ns);
		for (int j=0; j<LABEL_SIZE; j++){
			yhat[j] = a[ns.dza_pstn[ns.L-1]+j];
		}
		if (argmax(yhat) == label[i]) ctr += 1;
	}
	return ((float) ctr/num_images);
}

void log_train_progress(float train_image[][IMAGE_SIZE], int train_label[NUM_TRAIN],
 float test_image[][IMAGE_SIZE], int test_label[NUM_TEST], int epoch, float* weights,
 float* biases, int output_type, struct network_structure ns){

	float acc_train = evaluate(train_image, NUM_TRAIN, train_label, weights, biases, output_type, ns);
	float acc_test = evaluate(test_image, NUM_TEST, test_label, weights, biases, output_type, ns);
	printf("Epoch %d: Train %0.5f, Test %0.5f\n", epoch, acc_train, acc_test);
}

void delta_cross_entropy(float* deltas, float* a_list, int label){
	for (int i=0; i<LABEL_SIZE; i++){
		float y = 0.0; if (i == label) y = 1.0;
		deltas[i] = a_list[i] - y;
	}
}

void forward_back_prop(float image[IMAGE_SIZE], int label, float* weights,
 float* biases, float* deltas, float* a, float* z, int output_type,
 struct network_structure ns){
	// feedforward
	forward_prop(image, weights, biases, a, z, output_type, ns);

	// output error
	if (output_type){
		delta_cross_entropy(&deltas[ns.dza_pstn[ns.L-1]], &a[ns.dza_pstn[ns.L-1]], label);
	}
	else {
		for (int i=0; i<LABEL_SIZE; i++){
			float y = 0.0; if (i == label) y = 1.0;
			deltas[ns.dza_pstn[ns.L-1] + i] = (a[ns.dza_pstn[ns.L-1] + i] - y) *
			 sigmoid_prime(z[ns.dza_pstn[ns.L-1] + i]);
		}
	}

	// back propagate
	for (int l=ns.L-2; l>-1; l--){
		float wxd[ns.layers[l]];
		weight_x_d(&weights[ns.weights_pstn[l]], &deltas[ns.dza_pstn[l+1]],
		 wxd, ns.layers[l], ns.layers[l+1]);
		for (int i=0; i<ns.layers[l]; i++){
			deltas[ns.dza_pstn[l] + i] = wxd[i] * sigmoid_prime(z[ns.dza_pstn[l] + i]);
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
        float sigma = 1.0f / (sqrtf((float)ns.layers[l]));
        for(int i = 0; i < ns.layers[l]; i++) {
            for(int j = 0; j < ns.layers[l+1]; j++) {
                weights[ns.weights_pstn[l] + i*ns.layers[l+1] + j] = rand_normal(0.0f, sigma);
            }
        }
    }
	int batch[nb]; // OR: int *batch = (int*) calloc(nb, sizeof(int));

	// initialize timer
	float totalTime = 0.0;
	float avgTime = 0.0;

	// train NN
	for (int epoch=0; epoch<ne; epoch++){
		// output progresses
		if (epoch % log_interval == 0){
			log_train_progress(train_image, train_label, test_image, test_label, epoch,
			weights, biases, output_type, ns);
			if (epoch != 0) printf("  Agerage time per epoch: %f sec\n", avgTime);
		}

		// prepare batch
		for (int s=0; s<nb; s++){
			float x = ((float)rand()/RAND_MAX) * NUM_TRAIN;
			batch[s] = (int) x;
		}

		/* --- main part --- */
		clock_t start = clock();

		// feedforward & backpropagate
		for (int s=0; s<nb; s++){
			forward_back_prop(train_image[batch[s]], train_label[batch[s]],
			 weights, biases, &deltas[ns.num_nodes*s], &a_list[ns.num_nodes*s],
			 &z_list[ns.num_nodes*s], output_type, ns);
		}

		// gradient descent
		for (int l=0; l<ns.L-1; l++){
			for (int s=0; s<nb; s++){
				for (int i=0; i<ns.layers[l]; i++){
					biases[ns.biases_pstn[l]+i] -= (alpha/nb) * deltas[s*ns.num_nodes+ns.dza_pstn[l+1]+i];
					for (int j=0; j<ns.layers[l+1]; j++){
						weights[ns.weights_pstn[l]+i*ns.layers[l+1]+j] -=
						 (alpha/nb) * deltas[s*ns.num_nodes+ns.dza_pstn[l+1]+j] *
						 a_list[s*ns.num_nodes+ns.dza_pstn[l]+i];
					}
				}
			}
		}
		clock_t end = clock();
		float tElapsed = (float) (end - start) / CLOCKS_PER_SEC;
		if (epoch > 0) { // First iter is warm up
			totalTime += tElapsed;
			avgTime = totalTime / (float)(epoch-1);
		}
	}

	// output final result
	log_train_progress(train_image, train_label, test_image, test_label, ne,
	 weights, biases, output_type, ns);
	printf("  Agerage time per epoch: %f sec\n", avgTime);

	free(weights); free(biases); free(deltas); free(z_list); free(a_list);
	return 0;
}
