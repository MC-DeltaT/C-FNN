#include <assert.h>             // assert
#include "matrix.h"             // matrix & assoc.
#include "mv_ops.h"
#include <stddef.h>             // size_t
#include "vector.h"             // vector & assoc.


void mv_mul(matrix mat, vector vec, vector res)
{
    assert(mat.rows == res.size);
    assert(mat.columns == vec.size);

    for (size_t i = 0; i < mat.rows; ++i) {
        v_store(res, i, 0.0);
        for (size_t j = 0; j < mat.columns; ++j) {
            *v_access(res, i) += m_load(mat, i, j) * v_load(vec, j);
        }
    }
}

void mv_tmul(matrix mat, vector vec, vector res)
{
    assert(mat.rows == vec.size);
    assert(mat.columns == res.size);

    for (size_t j = 0; j < mat.columns; ++j) {
        v_store(res, j, 0.0);
        for (size_t i = 0; i < mat.rows; ++i) {
            *v_access(res, j) += m_load(mat, i, j) * v_load(vec, i);
        }
    }
}

void v_mmul(vector vec1, vector vec2, matrix res)
{
    assert(vec1.size == res.rows);
    assert(vec2.size == res.columns);

    for (size_t i = 0; i < res.rows; ++i) {
        for (size_t j = 0; j < res.columns; ++j) {
            m_store(res, i, j, v_load(vec1, i) * v_load(vec2, j));
        }
    }
}
