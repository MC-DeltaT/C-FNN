#ifndef MATH_H
#define MATH_H


// Performs the calculation sigmoid(x) = e^x / (e^x + 1).
double sigmoidf(double x);

// Derivative of the sigmoid function.
double sigmoid_primef(double x);

// Generates a random double with a standard normal distribution.
double standard_normal_randf(void);


#endif
