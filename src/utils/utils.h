#ifndef UTILS
#define UTILS

#include <stdlib.h>
#include <math.h>

long double random_init(long double l, long double u);
long double sigmoid(long double x);
long double sigmoid_derivative(long double x);
long double loss(long double output, long double expected_output);
long double loss_derivative(long double output, long double expected_output);
long double** arr2d_to_pp(size_t rows, size_t cols, long double arr[rows][cols]);

#endif
