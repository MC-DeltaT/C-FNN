#include "mnist_loader.h"
#include "network_training.h"       // training_data_point, training_dataset
#include <stdint.h>                 // uint32_t
#include <stdio.h>                  // file stuff, fprintf, size_t
#include <stdlib.h>                 // exit
#include "vector.h"                 // vector & assoc.
#include "vector_ops.h"             // v_fill


// Parses a uint32_t from 4 unsigned chars arranged from MSB to LSB.
static uint32_t parse_uint32_big_endian(unsigned char buf[4])
{
    return (uint32_t)buf[0] << 24 | (uint32_t)buf[1] << 16 | (uint32_t)buf[2] << 8 | (uint32_t)buf[3];
}


training_dataset load_mnist_dataset(char const* labels_filename, char const* images_filename)
{
    FILE* labels_file = fopen(labels_filename, "rb");
    if (!labels_file) {
        fprintf(stderr, "MNIST labels file open failed.");
        exit(EXIT_FAILURE);
    }

    FILE* images_file = fopen(images_filename, "rb");
    if (!images_file) {
        fprintf(stderr, "MNIST images file open failed.");
        exit(EXIT_FAILURE);
    }

    unsigned char buf[MNIST_INPUT_SIZE];

    // Check the magic number.
    if (4 != fread(buf, 1, 4, labels_file)) {
        fprintf(stderr, "MNIST labels file read failed.");
        exit(EXIT_FAILURE);
    }
    if (2049 != parse_uint32_big_endian(buf)) {
        fprintf(stderr, "MNIST labels file magic number incorrect.");
        exit(EXIT_FAILURE);
    }

    // Check the magic number.
    if (4 != fread(buf, 1, 4, images_file)) {
        fprintf(stderr, "MNIST images file read failed.");
        exit(EXIT_FAILURE);
    }
    if (2051 != parse_uint32_big_endian(buf)) {
        fprintf(stderr, "MNIST images file magic number incorrect.");
        exit(EXIT_FAILURE);
    }

    // Read the number of labels.
    if (4 != fread(buf, 1, 4, labels_file)) {
        fprintf(stderr, "MNIST labels file read failed.");
        exit(EXIT_FAILURE);
    }
    uint32_t num_labels = parse_uint32_big_endian(buf);

    // Read the number of images.
    if (4 != fread(buf, 1, 4, images_file)) {
        fprintf(stderr, "MNIST images file read failed.");
        exit(EXIT_FAILURE);
    }
    uint32_t num_images = parse_uint32_big_endian(buf);

    if (num_labels != num_images) {
        fprintf(stderr, "MNIST labels and images count mismatch.");
        exit(EXIT_FAILURE);
    }

    // Read the image dimensions.
    if (4 != fread(buf, 1, 4, images_file)) {
        fprintf(stderr, "MNIST images file read failed.");
        exit(EXIT_FAILURE);
    }
    uint32_t image_rows = parse_uint32_big_endian(buf);

    if (4 != fread(buf, 1, 4, images_file)) {
        fprintf(stderr, "MNIST images file read failed.");
        exit(EXIT_FAILURE);
    }
    uint32_t image_cols = parse_uint32_big_endian(buf);

    if (image_rows != MNIST_INPUT_ROWS) {
        fprintf(stderr, "MNIST image row count unexpected.");
        exit(EXIT_FAILURE);
    }
    if (image_cols != MNIST_INPUT_COLS) {
        fprintf(stderr, "MNIST image column count unexpected.");
        exit(EXIT_FAILURE);
    }

    training_dataset dataset;
    if (!(dataset.data_points = malloc(num_images * sizeof(training_data_point)))) {
        fprintf(stderr, "MNIST training data allocation failed.");
        exit(EXIT_FAILURE);
    }
    dataset.size = num_images;

    for (size_t n = 0; n < num_images; ++n) {
        unsigned char label;
        if (1 != fread(&label, 1, 1, labels_file)) {
            fprintf(stderr, "MNIST labels file read failed.");
            exit(EXIT_FAILURE);
        }

        if (label > 9) {
            fprintf(stderr, "MNIST label value invalid.");
            exit(EXIT_FAILURE);
        }

        if (MNIST_INPUT_SIZE != fread(buf, 1, MNIST_INPUT_SIZE, images_file)) {
            fprintf(stderr, "MNIST images file read failed.");
            exit(EXIT_FAILURE);
        }

        dataset.data_points[n].input = create_vector(MNIST_INPUT_SIZE);
        dataset.data_points[n].desired = create_vector(MNIST_OUTPUT_SIZE);

        for (size_t i = 0; i < MNIST_INPUT_SIZE; ++i) {
            v_store(dataset.data_points[n].input, i, (double)buf[i] / 255.0);
        }

        v_fill(dataset.data_points[n].desired, 0.0);
        v_store(dataset.data_points[n].desired, label, 1.0);
    }

    fclose(labels_file);
    labels_file = NULL;
    fclose(images_file);
    images_file = NULL;

    return dataset;
}

void unload_mnist_dataset(training_dataset* dataset)
{
    for (size_t i = 0; i < dataset->size; ++i) {
        destroy_vector(&dataset->data_points[i].input);
        destroy_vector(&dataset->data_points[i].desired);
    }

    free(dataset->data_points);
    dataset->data_points = NULL;

    dataset->size = 0;
}
