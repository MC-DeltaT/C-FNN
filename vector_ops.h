#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H


#include "vector.h"         // vector


// Vector-vector addition.
void v_add(vector vec1, vector vec2, vector res);

// Vector copy.
void v_cpy(vector src, vector dst);

// Gets the index of the element with the highest value.
size_t v_emaxi(vector vec);

// Fills vector elements with a value.
void v_fill(vector vec, double val);

// Applies a function to the elements of a vector.
void v_func(vector vec, double (*func)(double), vector res);

// Fills vector elements with the results of calls to a function.
void v_gen(vector vec, double (*func)(void));

// Vector-vector Hadamard product.
void v_hprod(vector vec1, vector vec2, vector res);

// Combined vector-scalar multiplication and vector-vector addition.
void v_sma(vector vec1, double val, vector vec2, vector res);

// Vector-scalar multiplication.
void v_smul(vector vec, double val, vector res);

// Vector-vector subtraction.
void v_sub(vector vec1, vector vec2, vector res);


#endif
