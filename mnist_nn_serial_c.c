/*
 * A complete serial implementation of a feedforward neural network in C.
 * Trains on the MNIST dataset using mini-batch gradient descent.
 *
 * Input Example:
 * ./serial 2 30 30 64 0.1 1 1
 * (2 hidden layers, 30 neurons each, 30 epochs, batch size 64, learning rate 0.1, softmax output, log every 1 epoch)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // Added for memset
#include "load_data.h"

#define IMAGE_SIZE 784 // 28*28
#define LABEL_SIZE 10
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Represents the network's structure and memory layout
struct network_structure {
    int layers[22];       // Number of neurons in each layer
    int weights_pstn[22]; // Start position for weights of each layer in the 1D array
    int biases_pstn[22];  // Start position for biases of each layer
    int dza_pstn[22];     // Start position for deltas/z/activations of each layer
    int num_nodes;        // Total number of neurons (excluding input layer)
    int num_weights;      // Total number of weights
    int num_biases;       // Total number of biases
    int L;                // Total number of layers
};

// --- Helper Functions ---

float rand_uniform(float a, float b) {
    float x = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return (b - a) * x + a;
}

float rand_normal(float mu, float sigma) {
    float z = sqrtf(-2.0f * logf(rand_uniform(0.0f, 1.0f))) *
              sinf(2.0f * M_PI * rand_uniform(0.0f, 1.0f));
    return mu + sigma * z;
}

int argmax(float* y) {
    float maxval = -1.0f;
    int maxidx = -1;
    for (int i = 0; i < LABEL_SIZE; i++) {
        if (y[i] > maxval) {
            maxval = y[i];
            maxidx = i;
        }
    }
    return maxidx;
}

// --- Core Neural Network Math ---

void weight_x_a(float* weights, float* a, float* wxa, int l_num, int l_next_num) {
    for (int i = 0; i < l_next_num; i++) {
        float tmp = 0.0f;
        for (int j = 0; j < l_num; ++j)
            tmp += weights[i + j * l_next_num] * a[j];
        wxa[i] = tmp;
    }
}

void weight_x_d(float* weights, float* d, float* wxd, int l_num, int l_next_num) {
    for (int i = 0; i < l_num; i++) {
        float tmp = 0.0f;
        for (int j = 0; j < l_next_num; ++j)
            tmp += weights[l_next_num * i + j] * d[j];
        wxd[i] = tmp;
    }
}

float sigmoid(float z) {
    return (1.0f / (1.0f + expf(-z)));
}

float sigmoid_prime(float z) {
    return sigmoid(z) * (1.0f - sigmoid(z));
}

float relu(float z) {
    return fmaxf(0.0f, z);
}

float relu_prime(float z) {
    return (z > 0.0f) ? 1.0f : 0.0f;
}

void softmax(float* a, float* z, int l) {
    float sum = 0.0f;
    float max_z = z[0];
    for (int j = 1; j < l; j++) {
        if (z[j] > max_z) max_z = z[j];
    }
    for (int j = 0; j < l; j++) {
        a[j] = expf(z[j] - max_z); // Subtract max_z for numerical stability
        sum += a[j];
    }
    for (int j = 0; j < l; j++) {
        a[j] /= sum;
    }
}

// --- Forward and Backward Propagation ---

void forward_prop(float image[IMAGE_SIZE], float* weights, float* biases, float* a,
                  float* z, int output_type, struct network_structure ns) {

    for (int i = 0; i < IMAGE_SIZE; i++) a[i] = image[i];

    for (int l = 1; l < ns.L; l++) {
        float wxa[ns.layers[l]];
        weight_x_a(&weights[ns.weights_pstn[l - 1]], &a[ns.dza_pstn[l - 1]],
                   wxa, ns.layers[l - 1], ns.layers[l]);
        for (int j = 0; j < ns.layers[l]; j++) {
            z[ns.dza_pstn[l] + j] = wxa[j] + biases[ns.biases_pstn[l - 1] + j];
        }
        if ((l == ns.L - 1) && output_type) {
            softmax(&a[ns.dza_pstn[l]], &z[ns.dza_pstn[l]], ns.layers[l]);
        } else {
            for (int j = 0; j < ns.layers[l]; j++) {
                a[ns.dza_pstn[l] + j] = relu(z[ns.dza_pstn[l] + j]);
            }
        }
    }
}

void delta_cross_entropy(float* deltas, float* a_list, int label) {
    for (int i = 0; i < LABEL_SIZE; i++) {
        float y = 0.0f;
        if (i == label) y = 1.0f;
        deltas[i] = a_list[i] - y;
    }
}

void forward_back_prop(float image[IMAGE_SIZE], int label, float* weights,
                       float* biases, float* deltas, float* a, float* z, int output_type,
                       struct network_structure ns) {
    // Feedforward
    forward_prop(image, weights, biases, a, z, output_type, ns);

    // Output error (delta for the last layer)
    if (output_type) {
        delta_cross_entropy(&deltas[ns.dza_pstn[ns.L - 1]], &a[ns.dza_pstn[ns.L - 1]], label);
    } else {
        for (int i = 0; i < LABEL_SIZE; i++) {
            float y = 0.0f;
            if (i == label) y = 1.0f;
            deltas[ns.dza_pstn[ns.L - 1] + i] = (a[ns.dza_pstn[ns.L - 1] + i] - y) *
                                               sigmoid_prime(z[ns.dza_pstn[ns.L - 1] + i]);
        }
    }

    // Backpropagate the error
    for (int l = ns.L - 2; l > 0; l--) { // Note: Loop goes to l > 0
        float wxd[ns.layers[l]];
        weight_x_d(&weights[ns.weights_pstn[l]], &deltas[ns.dza_pstn[l + 1]],
                   wxd, ns.layers[l], ns.layers[l + 1]);
        for (int i = 0; i < ns.layers[l]; i++) {
            deltas[ns.dza_pstn[l] + i] = wxd[i] * relu_prime(z[ns.dza_pstn[l] + i]);
        }
    }
}

// --- Evaluation and Logging ---

float evaluate(float image[][IMAGE_SIZE], int num_images, int label[],
               float* weights, float* biases, int output_type, struct network_structure ns) {
    int ctr = 0;
    float yhat[LABEL_SIZE];
    float* z = (float*) calloc(ns.num_nodes, sizeof(float));
    float* a = (float*) calloc(ns.num_nodes, sizeof(float));

    for (int i = 0; i < num_images; i++) {
        forward_prop(image[i], weights, biases, a, z, output_type, ns);
        for (int j = 0; j < LABEL_SIZE; j++) {
            yhat[j] = a[ns.dza_pstn[ns.L - 1] + j];
        }
        if (argmax(yhat) == label[i]) ctr += 1;
    }
    free(z);
    free(a);
    return ((float)ctr / num_images);
}

void log_train_progress(float train_image[][IMAGE_SIZE], int train_label[],
                        float test_image[][IMAGE_SIZE], int test_label[], int epoch, float* weights,
                        float* biases, int output_type, struct network_structure ns) {
    // Note: Evaluating on the full training set is slow. For quick checks,
    // you might evaluate on a smaller subset.
    float acc_train = evaluate(train_image, 10000, train_label, weights, biases, output_type, ns); // Evaluate on subset
    float acc_test = evaluate(test_image, NUM_TEST, test_label, weights, biases, output_type, ns);
    printf("Epoch %d: Train Acc: %0.5f, Test Acc: %0.5f", epoch, acc_train, acc_test);
}


// --- Main Execution ---

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "Error: Not enough arguments provided.\n");
        fprintf(stderr, "Usage: %s <layers> <hidden_nodes> <epochs> <batch_size> <learning_rate> <output_type>\n", argv[0]);
        return 1;
    }

    int nl = atoi(argv[1]);
    int nh = atoi(argv[2]);
    int ne = atoi(argv[3]);
    int nb = atoi(argv[4]);
    float alpha = atof(argv[5]);
    int output_type = atoi(argv[6]);
    int log_interval = (argc > 7) ? atoi(argv[7]) : 1;
    if (nl >= 20) {
        printf("Set your number of layers less than 20\n");
        return 0;
    }

    printf("Inputs:\n  Layers: %d\n  Hidden Neurons: %d\n  Epochs: %d\n  Batch Size: %d\n  Learning Rate: %.3f\n  Output Type: %d (0: Sigmoid+MSE, 1: Softmax+CrossEntropy)\n\n",
           nl, nh, ne, nb, alpha, output_type);

    load_mnist();

    // --- Setup Network Structure ---
    struct network_structure ns;
    memset(&ns, 0, sizeof(struct network_structure));
    ns.layers[0] = IMAGE_SIZE;
    for (int i = 0; i < nl; i++) {
        ns.layers[i + 1] = nh;
    }
    ns.layers[nl + 1] = LABEL_SIZE;
    ns.L = nl + 2;

    for (int i = 1; i < ns.L; i++) {
        ns.weights_pstn[i] = ns.weights_pstn[i-1] + ns.layers[i-1] * ns.layers[i];
        ns.biases_pstn[i] = ns.biases_pstn[i-1] + ns.layers[i];
        ns.dza_pstn[i] = ns.dza_pstn[i-1] + ns.layers[i-1];
    }
    ns.num_weights = ns.weights_pstn[ns.L-1] + ns.layers[ns.L-2] * ns.layers[ns.L-1];
    ns.num_biases = ns.biases_pstn[ns.L-1] + ns.layers[ns.L-1];
    ns.num_nodes = ns.dza_pstn[ns.L-1] + ns.layers[ns.L-1];


    // --- Allocate Memory ---
    float* weights = (float*) malloc(ns.num_weights * sizeof(float));
    float* biases = (float*) calloc(ns.num_biases, sizeof(float));

    // **NEW**: Gradient accumulators for mini-batch
    float* nabla_w = (float*) calloc(ns.num_weights, sizeof(float));
    float* nabla_b = (float*) calloc(ns.num_biases, sizeof(float));

    float* deltas = (float*) calloc(ns.num_nodes, sizeof(float));
    float* z_list = (float*) calloc(ns.num_nodes, sizeof(float));
    float* a_list = (float*) calloc(ns.num_nodes, sizeof(float));

    // --- Initialize Weights ---
    srand(42);
    for (int l = 0; l < ns.L - 1; l++) {
        float sigma = sqrtf(2.0f / (float)ns.layers[l]); // He initialization is often better with ReLU/Sigmoid
        for (int i = 0; i < ns.layers[l] * ns.layers[l + 1]; i++) {
            weights[ns.weights_pstn[l] + i] = rand_normal(0.0f, sigma);
        }
    }

    int batch[nb];
    float totalTime = 0.0;

    // --- Train Network ---
    for (int epoch = 0; epoch < ne; epoch++) {
        clock_t start = clock();

        // **CORRECTED**: Zero out gradient accumulators for the new mini-batch
        memset(nabla_w, 0, ns.num_weights * sizeof(float));
        memset(nabla_b, 0, ns.num_biases * sizeof(float));

        // Create a random mini-batch
        for (int s = 0; s < nb; s++) {
            batch[s] = rand() % NUM_TRAIN;
        }

        // --- Accumulate Gradients over the Mini-Batch ---
        for (int s = 0; s < nb; s++) {
            forward_back_prop(train_image[batch[s]], train_label[batch[s]],
                              weights, biases, deltas, a_list, z_list, output_type, ns);

            // **CORRECTED**: Add this sample's gradient to the accumulators
            for (int l = 0; l < ns.L - 1; l++) {
                // Biases
                for (int i = 0; i < ns.layers[l + 1]; i++) {
                    nabla_b[ns.biases_pstn[l] + i] += deltas[ns.dza_pstn[l + 1] + i];
                }
                // Weights
                for (int i = 0; i < ns.layers[l]; i++) {
                    for (int j = 0; j < ns.layers[l + 1]; j++) {
                        nabla_w[ns.weights_pstn[l] + j + i * ns.layers[l + 1]] +=
                            a_list[ns.dza_pstn[l] + i] * deltas[ns.dza_pstn[l + 1] + j];
                    }
                }
            }
        }

        // --- Update Weights and Biases (Single Step) ---
        // **CORRECTED**: Apply the averaged gradient to update weights and biases
        float learn_rate_per_batch = alpha / (float)nb;
        for (int i = 0; i < ns.num_weights; i++) {
            weights[i] -= learn_rate_per_batch * nabla_w[i];
        }
        for (int i = 0; i < ns.num_biases; i++) {
            biases[i] -= learn_rate_per_batch * nabla_b[i];
        }

        clock_t end = clock();
        totalTime += (float)(end - start) / CLOCKS_PER_SEC;

        if (epoch % log_interval == 0) {
            log_train_progress(train_image, train_label, test_image, test_label, epoch,
                               weights, biases, output_type, ns);
            printf(" (Time for epoch: %.3fs)\n", (float)(end - start) / CLOCKS_PER_SEC);
        }
    }

    // --- Final Result ---
    printf("\n--- Final Performance ---\n");
    log_train_progress(train_image, train_label, test_image, test_label, ne,
                       weights, biases, output_type, ns);
    printf("\nAverage time per epoch: %.3f sec\n", totalTime / ne);

    // --- Free Memory ---
    free(weights);
    free(biases);
    free(nabla_w);
    free(nabla_b);
    free(deltas);
    free(z_list);
    free(a_list);
    return 0;
}