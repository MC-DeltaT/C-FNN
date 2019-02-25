#include <assert.h>                 // assert
#include "matrix.h"                 // matrix & assoc.
#include "network_training.h"
#include "sffbp_network.h"          // sffbp_network & operations
#include "vector.h"                 // vector
#include "vector_ops.h"             // v_fma


// Updates the weights and biases using delta_weights and delta_biases, respectively, as part of gradient descent.
static void gradient_descent_update_sffbpn(sffbp_network network, double learning_rate)
{
    assert(network.size >= 1);

    sffbp_layer* layers = network.layers;

    for (size_t i = 1; i < network.size; ++i) {
        v_sma(layers[i].delta_biases, -learning_rate, layers[i].biases, layers[i].biases);
    }

    for (size_t i = 1; i < network.size; ++i) {
        v_sma(matrix_to_vector(layers[i].delta_weights), -learning_rate, matrix_to_vector(layers[i].weights), matrix_to_vector(layers[i].weights));
    }
}


void gradient_descent_sffbpn(sffbp_network network, training_dataset training_data, double learning_rate, void (*cost_function_deriv)(vector output, vector desired, vector result))
{
    for (size_t i = 0; i < training_data.size; ++i) {
        feedforward_sffbpn(network, training_data.data_points[i].input);
        backpropagate_sffbpn(network, training_data.data_points[i].desired, cost_function_deriv);
        gradient_descent_update_sffbpn(network, learning_rate);
    }
}
