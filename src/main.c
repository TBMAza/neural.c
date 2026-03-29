#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>

#include "core/neuron.h"
#include "core/layer.h"
#include "core/network.h"
#include "utils/utils.h"

int main(void) {
	srand(time(0));

	long double xor_in[4][2] = {
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0}
	};
	long double xor_out[4][1] = {
		{0.0},
		{1.0},
		{1.0},
		{0.0}
	};

	long double** xor_inputs = arr2d_to_pp(4, 2, xor_in);
	long double** xor_outputs = arr2d_to_pp(4, 1, xor_out);

	Network n;
	init_network(&n, 2, 4, (size_t[]){4, 4, 4, 1});
	long double loss_sigma = 0.0;
	for(size_t i = 0; i < 4; ++i) {
		network_output(&n, xor_inputs[i]);
		loss_sigma += loss(n.outputs[0], xor_outputs[i][0]);
		printf("input: {%Lf, %Lf}\texpected output: %Lf\toutput: %Lf\n", xor_inputs[i][0], xor_inputs[i][1], xor_outputs[i][0], n.outputs[0]);
	}
	printf("total loss: %Lf\n", loss_sigma/4);

	train_network(&n, 1e6, 1e-2, 4, xor_inputs, xor_outputs);

	loss_sigma = 0.0;
	for(size_t i = 0; i < 4; ++i) {
		network_output(&n, xor_inputs[i]);
		loss_sigma += loss(n.outputs[0], xor_outputs[i][0]);
		printf("input: {%Lf, %Lf}\texpected output: %Lf\toutput: %Lf\n", xor_inputs[i][0], xor_inputs[i][1], xor_outputs[i][0], n.outputs[0]);
	}
	printf("total loss: %Lf\n", loss_sigma/4);

	free_network(&n);
	for(size_t i = 0; i < 4; ++i) {
		free(xor_inputs[i]);
		free(xor_outputs[i]);
	}
	free(xor_inputs);
	free(xor_outputs);

	return 0;
}