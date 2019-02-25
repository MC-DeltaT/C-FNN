#include <assert.h>         // assert
#include "matrix.h"
#include <stdio.h>          // fprintf
#include <stdlib.h>         // exit, free, malloc, size_t
#include "vector.h"         // vector


matrix const null_matrix = {.columns = 0, .data = NULL, .rows = 0};


matrix create_matrix(size_t rows, size_t columns)
{
    matrix mat;

    if (rows == 0 || columns == 0) {
        mat.data = NULL;
    }
    else {
        if (!(mat.data = malloc(rows * columns * sizeof(double)))) {
            fprintf(stderr, "matrix data allocation failed.");
            exit(EXIT_FAILURE);
        }
    }

    mat.rows = rows;
    mat.columns = columns;

    return mat;
}

void destroy_matrix(matrix* mat)
{
    free(mat->data);
    mat->data = NULL;
    mat->rows = 0;
    mat->columns = 0;
}

vector matrix_to_vector(matrix mat)
{
    vector vec;
    vec.data = mat.data;
    vec.size = mat.rows * mat.columns;

    return vec;
}

double* m_access(matrix mat, size_t i, size_t j)
{
    assert(i < mat.rows);
    assert(j < mat.columns);

    return mat.data + (i * mat.columns) + j;
}

double m_load(matrix mat, size_t i, size_t j)
{
    return *m_access(mat, i, j);
}

void m_store(matrix mat, size_t i, size_t j, double val)
{
    *m_access(mat, i, j) = val;
}
