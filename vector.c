#include <assert.h>
#include <stdio.h>          // fprintf
#include <stdlib.h>         // exit, free, malloc, size_t
#include "vector.h"


vector const null_vector = {.data = NULL, .size = 0};


vector create_vector(size_t size)
{
    vector vec;

    if (size == 0) {
        vec.data = NULL;
    }
    else {
        if (!(vec.data = malloc(size * sizeof(double)))) {
            fprintf(stderr, "vector data allocation failed.");
            exit(EXIT_FAILURE);
        }
    }

    vec.size = size;

    return vec;
}

void destroy_vector(vector* vec)
{
    free(vec->data);
    vec->data = NULL;
    vec->size = 0;
}

double* v_access(vector vec, size_t i)
{
    assert(i < vec.size);

    return vec.data + i;
}

double v_load(vector vec, size_t i)
{
    return *v_access(vec, i);
}

void v_store(vector vec, size_t i, double val)
{
    *v_access(vec, i) = val;
}
