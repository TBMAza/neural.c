#ifndef NETWORK
#define NETWORK

#include "layer.h"

typedef struct {
    size_t layers_no;
    Layer* layers;
    long double* outputs;
} Network;

void init_network(Network* network, size_t inputs_no, size_t layers_no, size_t* neurons_per_layer);
void print_network(Network* network, char* indentation);
void network_output(Network* network, long double* inputs);
void train_network_step(Network* network, long double learning_rate);
void free_network(Network* network);
void backpropagate(Network* network, long double* training_outputs);
void train_network_on_single_io(Network* network, long double learning_rate, long double* training_inputs, long double* training_outputs);
void train_network_on_multiple_io(Network* network, long double learning_rate, size_t io_len, long double** training_inputs, long double** training_outputs);
void train_network(Network* network, size_t epochs, long double learning_rate, size_t io_len, long double** training_inputs, long double** training_outputs);

#endif