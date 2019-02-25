#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H


#include "network_training.h"       // training_data_point, training_dataset


#define MNIST_INPUT_ROWS 28
#define MNIST_INPUT_COLS 28
#define MNIST_INPUT_SIZE (MNIST_INPUT_ROWS * MNIST_INPUT_COLS)
#define MNIST_OUTPUT_SIZE 10


// Loads MNIST images and labels from the specified files as training data.
training_dataset load_mnist_dataset(char const* labels_filename, char const* images_filename);

// Frees the training data resources acquired from load_mnist_dataset.
void unload_mnist_dataset(training_dataset* dataset);


#endif
