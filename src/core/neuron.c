#include "neuron.h"
#include "../utils/utils.h"

#include <stdio.h>

void init_neuron(Neuron* neuron, size_t inputs_no) {
	if(inputs_no == 0) {
		inputs_no = 1;
	}

	neuron->inputs_no = inputs_no;
	neuron->inputs = NULL;
	neuron->weights = malloc(sizeof(long double) * inputs_no);
	for(size_t i = 0; i < inputs_no; ++i) {
		neuron->weights[i] = random_init(0.0, 1.0);
	}
	neuron->bias = random_init(0.0, 1.0);
	neuron->output = 0.0;
	neuron->delta = 0.0;
}
void print_neuron(Neuron* neuron, char* indentation) {
	if(!neuron) {
		return;
	}
	if(!indentation) {
		indentation = "";
	}
	long double* placeholder = calloc(neuron->inputs_no, sizeof(long double));
	if(!neuron->inputs) {
		neuron->inputs = placeholder;
	}
	printf("%s{\n", indentation);
	printf(
			"%s\t.inputs_no: %zu,\n"
			"%s\t.inputs: {\n",
			indentation,
			neuron->inputs_no,
			indentation
	);
	for(size_t i = 0; i < neuron->inputs_no; ++i) {
		printf("%s\t\t%Lf,\n", indentation, neuron->inputs[i]);
	}
	printf("%s\t},\n\t%s.weights: {\n", indentation, indentation);
	for(size_t i = 0; i < neuron->inputs_no; ++i) {
		printf("%s\t\t%Lf,\n", indentation, neuron->weights[i]);
	}
	printf(
			"%s\t},\n\t%s.bias: %Lf,\n"
			"%s\t.output: %Lf,\n"
			"%s\t.delta: %Lf\n",
			indentation,
			indentation,
			neuron->bias,
			indentation,
			neuron->output,
			indentation,
			neuron->delta
	);
	printf("%s}\n", indentation);
	if(neuron->inputs == placeholder) {
		neuron->inputs = NULL;
	}
	free(placeholder);
}
void neuron_output(Neuron* neuron, long double* inputs) {
	if(!(neuron && inputs)) {
		return;
	}
	neuron->inputs = inputs;
	neuron->output = neuron->bias;
	for(size_t i = 0; i < neuron->inputs_no; ++i) {
		neuron->output += neuron->inputs[i] * neuron->weights[i];
	}
	neuron->output = sigmoid(neuron->output);
}
void train_neuron_step(Neuron* neuron, long double learning_rate) {
	neuron->bias -= learning_rate * neuron->delta;
	for(int i = 0; i < neuron->inputs_no; ++i) {
		neuron->weights[i] -= learning_rate * neuron->delta * neuron->inputs[i];
	}
}
void free_neuron(Neuron* neuron) {
	/* neuron->inputs is not owned by the neuron — it points into the
	   previous layer's outputs array, which that layer is responsible
	   for freeing. Only free what we allocated: weights. */
	free(neuron->weights);
}