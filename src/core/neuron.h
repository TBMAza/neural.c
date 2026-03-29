#ifndef NEURON
#define NEURON

#include <stddef.h>

typedef struct {
	size_t inputs_no;

	long double* inputs;
	long double* weights;
	long double bias;
	long double output;
	
	long double delta;
} Neuron;

void init_neuron(Neuron* neuron, size_t inputs_no);
void print_neuron(Neuron* neuron, char* indentation);
void neuron_output(Neuron* neuron, long double* inputs);
void train_neuron_step(Neuron* neuron, long double learning_rate);
void free_neuron(Neuron* neuron);

#endif
