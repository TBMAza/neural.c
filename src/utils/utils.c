#include "utils.h"
#include <stdlib.h>
#include <math.h>

long double random_init(long double l, long double u) {
	if(u <= l) {
		l = 0;
		u = 1;
	}
	return ((long double)rand()/RAND_MAX) * (u-l) + l;
}
long double sigmoid(long double x) {
	return 1.0 / (1.0 + exp(-x));
}
long double sigmoid_derivative(long double x) {
	return x * (1 - x);
}
long double loss(long double output, long double expected_output) {
	return pow(output - expected_output, 2);
}
long double loss_derivative(long double output, long double expected_output) {
	return 2 * (output - expected_output);
}
long double** arr2d_to_pp(size_t rows, size_t cols, long double arr[rows][cols]) {
	long double** pp = malloc(sizeof(long double*) * rows);
	for(size_t i = 0; i < rows; ++i) {
		pp[i] = malloc(sizeof(long double) * cols);
	}
	for(size_t i = 0; i < rows; ++i) {
		for(size_t j = 0; j < cols; ++j) {
			pp[i][j] = arr[i][j];
		}
	}
	return pp;
}