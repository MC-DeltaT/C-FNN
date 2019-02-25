#include "math.h"
#include <math.h>           // math stuff
#include <stdlib.h>         // rand


double sigmoidf(double x)
{
    double ex = exp(x);
    double res = ex / (1.0 + ex);

    return res;
}

double sigmoid_primef(double x)
{
    double ex = exp(x);
    double ex_p1 = ex + 1.0;

    double res = ex / (ex_p1 * ex_p1);

    return res;
}

double standard_normal_randf(void)
{
    double const pi = acos(-1.0);

    // Box-Muller method.

    double u1 = (double)rand() / (double)RAND_MAX;
    if (u1 == 0) {
        u1 = 1;
    }

    double u2 = (double)rand() / (double)RAND_MAX;
    if (u2 == 0) {
        u2 = 1;
    }

    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);

    return z;
}
