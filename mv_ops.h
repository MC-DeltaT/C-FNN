#ifndef MV_OPS_H
#define MV_OPS_H


#include "matrix.h"         // matrix
#include "vector.h"         // vector


// Matrix-vector multiplication.
void mv_mul(matrix mat, vector vec, vector res);

// Matrix-vector multiplication, with transposed matrix.
void mv_tmul(matrix mat, vector vec, vector res);

// Vector-vector matrix product.
void v_mmul(vector vec1, vector vec2, matrix res);


#endif
