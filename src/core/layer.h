#ifndef LAYER
#define LAYER

#include "neuron.h"

typedef struct {
	size_t neurons_no;
	Neuron* neurons;
	long double* outputs;
} Layer;

void init_layer(Layer* layer, size_t neurons_no, size_t inputs_per_neuron);
void print_layer(Layer* layer, char* indentation);
void layer_output(Layer* layer, long double* inputs);
void train_layer_step(Layer* layer, long double learning_rate);
void free_layer(Layer* layer);

#endif
