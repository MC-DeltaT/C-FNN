#include <assert.h>             // assert
#include "math.h"               // standard_normal_rand
#include "mnist_loader.h"       // MNIST stuff
#include "network_training.h"   // training_dataset, gradient_descent_sffbpn, etc
#include "sffbp_network.h"      // sffbp_layer, sffbp_network, etc
#include <stdio.h>              // getchar
#include <stdlib.h>             // free, malloc, size_t
#include <time.h>               // time
#include "vector.h"             // vector
#include "vector_ops.h"         // vector operations


// Performs the calculation cost = 0.5 * sum((activations - desired)^2).
double quadratic_cost(vector activations, vector expected)
{
    assert(activations.size == expected.size);

    double cost = 0.0;
    for (size_t i = 0; i < activations.size; ++i) {
        double diff = v_load(activations, i) - v_load(expected, i);
        cost += diff * diff;
    }

    return 0.5 * cost;
}

// Derivative of the quadratic cost function.
void quadratic_cost_prime(vector activations, vector desired, vector result)
{
    assert(activations.size == desired.size);
    assert(activations.size == result.size);

    v_sub(activations, desired, result);
}


// Initialises the buffers in a layer with the appropriate values.
void init_layer(sffbp_layer layer)
{
    v_fill(layer.activations, 0.0);
    v_gen(layer.biases, &standard_normal_randf);
    v_fill(layer.buffer, 0.0);
    v_fill(layer.delta_activations, 0.0);
    v_fill(layer.delta_biases, 0.0);
    v_fill(matrix_to_vector(layer.delta_weights), 0.0);
    v_fill(layer.preactivations, 0.0);
    v_gen(matrix_to_vector(layer.weights), &standard_normal_randf);
}


void evaluate_network(sffbp_network network, training_dataset evaluation_data)
{
    size_t correct = 0;
    size_t tested = 0;
    double cost = 0;

    for (size_t i = 0; i < evaluation_data.size; i += 10) {
        feedforward_sffbpn(network, evaluation_data.data_points[i].input);

        vector activations = network.layers[network.size - 1].activations;
        vector desired = evaluation_data.data_points[i].desired;

        cost += quadratic_cost(activations, desired);
        size_t label = v_emaxi(activations);
        size_t correct_label = v_emaxi(desired);
        if (label == correct_label) {
            ++correct;
        }
        ++tested;
    }

    printf("Accuracy: %lf  |  Average cost: %lf\n", (double)correct / (double)tested, cost / (double)tested);
}

#define NUM_LAYERS 3

int main(void)
{
    printf("Press enter to begin.\n");
    getchar();

    srand(time(NULL));

    size_t const layer_sizes[NUM_LAYERS] = {MNIST_INPUT_SIZE, 50, MNIST_OUTPUT_SIZE};
    double const learning_rate = 0.6;

    sffbp_network network;
    network.layers = malloc(NUM_LAYERS * sizeof(sffbp_layer));
    network.size = NUM_LAYERS;

    network.layers[0] = create_input_sffbp_layer(layer_sizes[0]);

    for (size_t i = 1; i < NUM_LAYERS; ++i) {
        network.layers[i] = create_sffbp_layer(layer_sizes[i], layer_sizes[i - 1]);
        init_layer(network.layers[i]);
    }

    training_dataset mnist = load_mnist_dataset("mnist_training_labels.dat", "mnist_training_images.dat");
    //mnist.size = 100;

    printf("Training dataset size: %zd\n", mnist.size);
    printf("Learning rate: %lf\n", learning_rate);
    printf("\n");

    while (1) {
        gradient_descent_sffbpn(network, mnist, learning_rate, &quadratic_cost_prime);
        evaluate_network(network, mnist);
    }

    unload_mnist_dataset(&mnist);

    for (size_t i = 0; i < NUM_LAYERS; ++i) {
        destroy_sffbp_layer(&network.layers[i]);
    }

    free(network.layers);
    network.layers = NULL;
    network.size = 0;
}
