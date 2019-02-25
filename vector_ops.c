#include <assert.h>             // assert
#include <math.h>               // fmaf
#include <stddef.h>             // size_t
#include "vector.h"             // vector
#include "vector_ops.h"


void v_add(vector vec1, vector vec2, vector res)
{
    assert(vec1.size == vec2.size);
    assert(vec1.size == res.size);

    for (size_t i = 0; i < res.size; ++i) {
        v_store(res, i, v_load(vec1, i) + v_load(vec2, i));
    }
}

void v_cpy(vector src, vector dst)
{
    assert(src.size == dst.size);

    for (size_t i = 0; i < dst.size; ++i) {
        v_store(dst, i, v_load(src, i));
    }
}

size_t v_emaxi(vector vec)
{
    double max = -INFINITY;
    size_t index = 0;

    for (size_t i = 0; i < vec.size; ++i) {
        double e = v_load(vec, i);
        if (e > max) {
            max = e;
            index = i;
        }
    }

    return index;
}

void v_fill(vector vec, double val)
{
    for (size_t i = 0; i < vec.size; ++i) {
        v_store(vec, i, val);
    }
}

void v_func(vector vec, double (*func)(double), vector res)
{
    assert(vec.size == res.size);

    for (size_t i = 0; i < res.size; ++i) {
        v_store(res, i, func(v_load(vec, i)));
    }
}

void v_gen(vector vec, double (*func)(void))
{
    for (size_t i = 0; i < vec.size; ++i) {
        v_store(vec, i, func());
    }
}

void v_hprod(vector vec1, vector vec2, vector res)
{
    assert(vec1.size == vec2.size);
    assert(vec1.size == res.size);

    for (size_t i = 0; i < res.size; ++i) {
        v_store(res, i, v_load(vec1, i) * v_load(vec2, i));
    }
}

void v_sma(vector vec1, double val, vector vec2, vector res)
{
    assert(vec1.size == vec2.size);
    assert(vec1.size == res.size);

    for (size_t i = 0; i < res.size; ++i) {
        v_store(res, i, (v_load(vec1, i) * val) + v_load(vec2, i));
    }
}

void v_smul(vector vec, double val, vector res)
{
    assert(vec.size == res.size);

    for (size_t i = 0; i < res.size; ++i) {
        v_store(res, i, v_load(vec, i) * val);
    }
}

void v_sub(vector vec1, vector vec2, vector res)
{
    assert(vec1.size == vec2.size);
    assert(vec1.size == res.size);

    for (size_t i = 0; i < res.size; ++i) {
        v_store(res, i, v_load(vec1, i) - v_load(vec2, i));
    }
}
