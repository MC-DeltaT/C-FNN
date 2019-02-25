#ifndef MATRIX_H
#define MATRIX_H


#include <stddef.h>         // size_t
#include "vector.h"         // vector


// Represents a matrix of double.
typedef struct {
    size_t columns;
    double* data;
    size_t rows;
} matrix;


// matrix with null data and dimensions of 0.
extern matrix const null_matrix;


// Creates a matrix with the specified dimensions.
matrix create_matrix(size_t rows, size_t columns);

// Frees a matrix's resources.
void destroy_matrix(matrix* mat);

// Creates a vector viewing the given matrix's data.
vector matrix_to_vector(matrix mat);

// Gets a pointer to the element at the specified row and column.
double* m_access(matrix mat, size_t i, size_t j);

// Reads the element at the specified row and column.
double m_load(matrix mat, size_t i, size_t j);

// Writes to the element at the specfied row and column.
void m_store(matrix mat, size_t i, size_t j, double val);


#endif
