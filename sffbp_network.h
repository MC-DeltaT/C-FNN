#ifndef SFFBP_NETWORK_H
#define SFFBP_NETWORK_H


#include "matrix.h"             // matrix & assoc
#include <stddef.h>             // size_t
#include "vector.h"             // vector & assoc.


// Represents a feedforward, backpropagatable layer of sigmoid neurons.
typedef struct {
    vector activations;
    vector biases;
    vector buffer;
    vector delta_activations;
    vector delta_biases;
    matrix delta_weights;
    vector preactivations;
    size_t prev_size;
    size_t size;
    matrix weights;
} sffbp_layer;

// Represents a neural network of sffbp_layer.
typedef struct {
    sffbp_layer* layers;
    size_t size;
} sffbp_network;


// Backpropagates the rate of change of cost w.r.t the network parameters.
void backpropagate_sffbpn(sffbp_network network, vector desired_output, void (*cost_function_deriv)(vector output, vector desired, vector result));

// Creates an sffbp_layer for use as a hidden or output layer.
sffbp_layer create_sffbp_layer(size_t size, size_t prev_size);

// Creates an sffbp_layer for use as an input layer.
sffbp_layer create_input_sffbp_layer(size_t size);

// Frees a sffbp_layer's resources.
void destroy_sffbp_layer(sffbp_layer* layer);

// Feeds an input through a network.
void feedforward_sffbpn(sffbp_network network, vector input);


#endif
