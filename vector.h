#ifndef VECTOR_H
#define VECTOR_H


#include <stddef.h>         // size_t


// Represents a vector of double.
typedef struct {
    double* data;
    size_t size;
} vector;


// vector with null data and 0 size.
extern vector const null_vector;


// Creates a vector of the specified size.
vector create_vector(size_t size);

// Frees a vector's resources.
void destroy_vector(vector* vec);

// Gets a pointer to the element at the specified index.
double* v_access(vector vec, size_t i);

// Reads the element at the specified index.
double v_load(vector vec, size_t i);

// Writes to the element at the specified index.
void v_store(vector vec, size_t i, double val);


#endif
