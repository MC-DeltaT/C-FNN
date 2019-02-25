#ifndef NETWORK_TRAINING_H
#define NETWORK_TRAINING_H


#include "vector.h"             // vector
#include "sffbp_network.h"      // sffbp_network & operations
#include <stddef.h>             // size_t


// Represents one training data point.
typedef struct {
    vector desired;
    vector input;
} training_data_point;

// Array of training_data_point.
typedef struct {
    training_data_point* data_points;
    size_t size;
} training_dataset;


// Trains a sffbp_network with gradient descent.
void gradient_descent_sffbpn(sffbp_network network, training_dataset training_data, double learning_rate, void (*cost_function_deriv)(vector output, vector desired, vector result));


#endif
