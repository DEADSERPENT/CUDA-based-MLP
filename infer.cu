#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cassert>

#define IMAGE_SIZE 784 // 28*28
#define LABEL_SIZE 10
#define MODEL_PATH "./model_checkpoint.bin"
#define MAX_LAYERS 22

// Network structure
struct network_structure {
    int layers[MAX_LAYERS];
    int weights_pstn[MAX_LAYERS];
    int biases_pstn[MAX_LAYERS];
    int dza_pstn[MAX_LAYERS];
    int num_nodes;
    int L;
};

// Host functions for initialization
__host__ float relu(float z) {
    return fmaxf(0.0f, z);
}

__host__ void softmax(float* a, float* z, int l) {
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

__host__ int argmax(float* y) {
    float maxval = -1.0;
    int maxidx = -1;
    for (int i = 0; i < LABEL_SIZE; i++) {
        if (y[i] > maxval) {
            maxval = y[i];
            maxidx = i;
        }
    }
    return maxidx;
}

// Matrix multiplication for forward pass
__host__ void weight_x_a(float* weights, float* a, float* wxa, int l_num, int l_next_num) {
    for (int i = 0; i < l_next_num; i++) {
        float tmp = 0;
        for (int j = 0; j < l_num; ++j)
            tmp += weights[i + j * l_next_num] * a[j];
        wxa[i] = tmp;
    }
}

// Forward propagation
__host__ void forward_prop(float image[IMAGE_SIZE], float* weights, float* biases,
                          float* a, float* z, struct network_structure ns) {

    // Copy input to activation layer
    for (int i = 0; i < IMAGE_SIZE; i++)
        a[i] = image[i];

    // Process each layer
    for (int l = 1; l < ns.L; l++) {
        float wxa[IMAGE_SIZE];
        weight_x_a(&weights[ns.weights_pstn[l-1]], &a[ns.dza_pstn[l-1]],
                  wxa, ns.layers[l-1], ns.layers[l]);

        for (int j = 0; j < ns.layers[l]; j++) {
            z[ns.dza_pstn[l] + j] = wxa[j] + biases[ns.biases_pstn[l-1] + j];
        }

        // Apply activation function
        if (l == ns.L - 1) {
            // Output layer: softmax
            softmax(&a[ns.dza_pstn[l]], &z[ns.dza_pstn[l]], ns.layers[l]);
        } else {
            // Hidden layers: ReLU
            for (int j = 0; j < ns.layers[l]; j++) {
                a[ns.dza_pstn[l] + j] = relu(z[ns.dza_pstn[l] + j]);
            }
        }
    }
}

// Load model from checkpoint file
__host__ int load_model(const char* filepath, float* weights, float* biases,
                       int* nl, int* nh) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        printf("Error: Cannot open file %s for reading\n", filepath);
        return -1;
    }

    // Read header
    fread(nl, sizeof(int), 1, f);
    fread(nh, sizeof(int), 1, f);

    // Calculate sizes
    int num_weights = (784 * (*nh)) + ((*nh) * (*nh)) * ((*nl) - 1) + ((*nh) * 10);
    int num_biases = (*nh) * (*nl) + 10;

    // Read weights and biases
    fread(weights, sizeof(float), num_weights, f);
    fread(biases, sizeof(float), num_biases, f);

    fclose(f);
    return 0;
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    int verbose = 0;
    int benchmark = 0;
    const char* model_path = MODEL_PATH;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--benchmark") == 0) {
            benchmark = 1;
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("CUDA MNIST Inference\n");
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -v, --verbose    Print detailed information\n");
            printf("  -b, --benchmark  Show inference timing\n");
            printf("  -m, --model      Path to model checkpoint (default: ./model_checkpoint.bin)\n");
            printf("  -h, --help       Show this help message\n");
            printf("\nInput: Reads 784 space-separated floats from stdin\n");
            printf("Output: Prints predicted digit (0-9) to stdout\n");
            return 0;
        }
    }

    // Load model
    int nl, nh;
    int max_weights = (784 * 2048) + (2048 * 2048) * 20 + (2048 * 10);
    int max_biases = 2048 * 20 + 10;

    float* weights = (float*)malloc(max_weights * sizeof(float));
    float* biases = (float*)malloc(max_biases * sizeof(float));

    if (verbose) printf("Loading model from %s...\n", model_path);

    if (load_model(model_path, weights, biases, &nl, &nh) != 0) {
        printf("Failed to load model\n");
        free(weights);
        free(biases);
        return 1;
    }

    if (verbose) {
        printf("Model loaded successfully\n");
        printf("  Layers: %d (hidden)\n", nl);
        printf("  Neurons per layer: %d\n", nh);
    }

    // Initialize network structure
    struct network_structure ns;
    for (int i = 0; i < MAX_LAYERS; i++) {
        ns.weights_pstn[i] = 0;
        ns.biases_pstn[i] = 0;
        ns.dza_pstn[i] = 0;
        ns.layers[i] = 0;
    }

    ns.layers[0] = 784;
    for (int i = 0; i < nl; i++) {
        ns.layers[i + 1] = nh;
    }
    ns.layers[nl + 1] = 10;
    ns.num_nodes = 784 + nh * nl + 10;
    ns.L = nl + 2;

    for (int i = 0; i < ns.L - 2; i++) {
        ns.weights_pstn[i + 1] = ns.layers[i] * ns.layers[i + 1] + ns.weights_pstn[i];
        ns.biases_pstn[i + 1] = ns.layers[i + 1] + ns.biases_pstn[i];
    }
    for (int i = 0; i < ns.L - 1; i++) {
        ns.dza_pstn[i + 1] = ns.layers[i] + ns.dza_pstn[i];
    }

    // Allocate memory for inference
    float* z = (float*)calloc(ns.num_nodes, sizeof(float));
    float* a = (float*)calloc(ns.num_nodes, sizeof(float));
    float image[IMAGE_SIZE];

    if (verbose) printf("Reading input (784 values)...\n");

    // Read input from stdin
    for (int i = 0; i < IMAGE_SIZE; i++) {
        if (scanf("%f", &image[i]) != 1) {
            printf("Error: Failed to read input value %d\n", i);
            free(weights);
            free(biases);
            free(z);
            free(a);
            return 1;
        }
    }

    if (verbose) printf("Running inference...\n");

    // Time inference if benchmarking
    cudaEvent_t start, stop;
    float elapsed_time = 0;

    if (benchmark) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    // Run forward propagation
    forward_prop(image, weights, biases, a, z, ns);

    if (benchmark) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Get prediction
    int predicted = argmax(&a[ns.dza_pstn[ns.L - 1]]);

    // Output result
    if (verbose) {
        printf("\nPrediction: %d\n", predicted);
        if (benchmark) {
            printf("Inference time: %.4f ms\n", elapsed_time);
        }
        printf("\nOutput probabilities:\n");
        for (int i = 0; i < LABEL_SIZE; i++) {
            printf("  Digit %d: %.6f\n", i, a[ns.dza_pstn[ns.L - 1] + i]);
        }
    } else {
        printf("%d\n", predicted);
    }

    // Cleanup
    free(weights);
    free(biases);
    free(z);
    free(a);

    return 0;
}
