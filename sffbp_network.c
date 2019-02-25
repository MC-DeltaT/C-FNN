#include <assert.h>             // assert
#include "math.h"               // sigmoidf, sigmoid_primef
#include "matrix.h"             // matrix & assoc.
#include "mv_ops.h"             // matrix & vector hybrid operations
#include "sffbp_network.h"
#include "stddef.h"             // size_t
#include "vector.h"             // vector & assoc.
#include "vector_ops.h"         // vector operations


// Calculates delta_activations for an ouput_layer
static void backpropagate_output_sffbpl(sffbp_layer layer, vector desired_output, void (*cost_function_deriv)(vector output, vector desired, vector result))
{
    cost_function_deriv(layer.activations, desired_output, layer.delta_activations);
    v_func(layer.preactivations, &sigmoid_primef, layer.buffer);
    v_hprod(layer.delta_activations, layer.buffer, layer.delta_activations);
}

// Calculates delta_activations for layer.
static void backpropagate_sffbpl(sffbp_layer layer, sffbp_layer next_layer)
{
    mv_tmul(next_layer.weights, next_layer.delta_activations, layer.delta_activations);
    v_func(layer.preactivations, &sigmoid_primef, layer.buffer);
    v_hprod(layer.delta_activations, layer.buffer, layer.delta_activations);
}

// Calculates the delta_biases and delta_weights for layer.
static void calc_parameter_deltas_sffbpl(sffbp_layer layer, sffbp_layer prev_layer)
{
    // Bias deltas.
    v_cpy(layer.delta_activations, layer.delta_biases);

    // Weight deltas.
    v_mmul(layer.delta_activations, prev_layer.activations, layer.delta_weights);
}

// Calculates preactivations and activations for layer.
static void feedforward_sffbpl(sffbp_layer layer, sffbp_layer prev_layer)
{
    mv_mul(layer.weights, prev_layer.activations, layer.preactivations);
    v_add(layer.preactivations, layer.biases, layer.preactivations);
    v_func(layer.preactivations, &sigmoidf, layer.activations);
}


void backpropagate_sffbpn(sffbp_network network, vector desired_output, void (*cost_function_deriv)(vector output, vector desired, vector result))
{
    assert(network.size >= 1);

    backpropagate_output_sffbpl(network.layers[network.size - 1], desired_output, cost_function_deriv);

    for (size_t i = network.size - 1; i >= 2; --i) {
        backpropagate_sffbpl(network.layers[i - 1], network.layers[i]);
        calc_parameter_deltas_sffbpl(network.layers[i - 1], network.layers[i - 2]);
    }
}

sffbp_layer create_sffbp_layer(size_t size, size_t prev_size)
{
    sffbp_layer layer;

    layer.activations = create_vector(size);
    layer.biases = create_vector(size);
    layer.buffer = create_vector(size);
    layer.delta_activations = create_vector(size);
    layer.delta_biases = create_vector(size);
    layer.delta_weights = create_matrix(size, prev_size);
    layer.preactivations = create_vector(size);
    layer.prev_size = prev_size;
    layer.size = size;
    layer.weights = create_matrix(size, prev_size);

    return layer;
}

sffbp_layer create_input_sffbp_layer(size_t size)
{
    sffbp_layer layer;

    layer.activations = null_vector;//create_vector(size);
    layer.biases = null_vector;
    layer.buffer = null_vector;
    layer.delta_activations = null_vector;
    layer.delta_biases = null_vector;
    layer.delta_weights = null_matrix;
    layer.preactivations = null_vector;
    layer.prev_size = 0;
    layer.size = size;
    layer.weights = null_matrix;

    return layer;
}

void destroy_sffbp_layer(sffbp_layer* layer)
{
    destroy_vector(&layer->activations);
    destroy_vector(&layer->biases);
    destroy_vector(&layer->buffer);
    destroy_vector(&layer->delta_activations);
    destroy_vector(&layer->delta_biases);
    destroy_matrix(&layer->delta_weights);
    destroy_vector(&layer->preactivations);
    layer->prev_size = 0;
    layer->size = 0;
    destroy_matrix(&layer->weights);
}

void feedforward_sffbpn(sffbp_network network, vector input)
{
    assert(network.size >= 1);

    //v_cpy(network.layers[0].activations, input);
    network.layers[0].activations = input;

    for (size_t i = 0; i < network.size - 1; ++i) {
        feedforward_sffbpl(network.layers[i + 1], network.layers[i]);
    }
}
