#include "network.h"
#include "../utils/utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void init_network(Network* network, size_t inputs_no, size_t layers_no, size_t* neurons_per_layer) {
    if(!neurons_per_layer) {
        return;
    }
    if(inputs_no == 0) {
        inputs_no = 1;
    }
    if(layers_no == 0) {
        layers_no = 1;
    }
    network->layers_no = layers_no;
    network->outputs = NULL;
    network->layers = calloc(layers_no, sizeof(Layer));
    for(size_t i = 0; i < layers_no; ++i) {
        init_layer(network->layers + i, neurons_per_layer[i], inputs_no);
        inputs_no = network->layers[i].neurons_no;
    }
}
void print_network(Network* network, char* indentation) {
    if(!(network && network->layers)) {
        return;
    }
    if(!indentation) {
        indentation = "";
    }
    long double* placeholder = calloc(network->layers_no, sizeof(long double));
	if(!network->outputs) {
		network->outputs = placeholder;
	}
    printf("%s{\n", indentation);
    printf("%s\t.layers_no: %zu\n", indentation, network->layers_no);
    printf("%s\t.outputs: {", indentation);
    for(int i = 0; i < network->layers_no; ++i) {
        printf("%Lf, ", network->outputs[i]);
    }
    printf("}\n");
    printf("%s\t.layers:\n", indentation);

    char next_indentation[strlen(indentation) + 2];
    strcpy(next_indentation, indentation);
    strcat(next_indentation, "\t");
    for(int i = 0; i < network->layers_no; ++i) {
        print_layer(network->layers + i, next_indentation);
    }
    printf("%s}\n", indentation);
    if(network->outputs == placeholder) {
		network->outputs = NULL;
	}
	free(placeholder);
}
void network_output(Network* network, long double* inputs){
    if(!(network && inputs)) {
        return;
    }
    for(size_t i = 0; i < network->layers_no; ++i) {
        layer_output(network->layers + i, inputs);
        inputs = network->layers[i].outputs;
    }
    network->outputs = inputs;
}
void train_network_step(Network* network, long double learning_rate) {
    if(!network) {
        return;
    }
    for(size_t i = 0; i < network->layers_no; ++i) {
        train_layer_step(network->layers + i, learning_rate);
    }
}
void free_network(Network* network) {
    for(size_t i = 0; i < network->layers_no; ++i) {
        free_layer(network->layers + i);
    }
    free(network->layers);
}
void backpropagate(Network* network, long double* training_outputs) {
    for(long i = network->layers_no-1; i >= 0; --i) {
        if(i == network->layers_no-1) {
            for(size_t j = 0; j < network->layers[i].neurons_no; ++j) {
                network->layers[i].neurons[j].delta =
                    loss_derivative(network->layers[i].neurons[j].output, training_outputs[j]) *
                    sigmoid_derivative(network->layers[i].neurons[j].output);
            }
        }
        else {
            for(size_t j = 0; j < network->layers[i].neurons_no; ++j) {
                long double sigma_delta_x_weight_prev = 0.0;
                for(size_t k = 0; k < network->layers[i+1].neurons_no; ++k) {
                    sigma_delta_x_weight_prev += network->layers[i+1].neurons[k].delta * network->layers[i+1].neurons[k].weights[j];
                }
                network->layers[i].neurons[j].delta = sigma_delta_x_weight_prev * sigmoid_derivative(network->layers[i].neurons[j].output);
            }
        }
    }
}

void train_network_on_single_io(Network* network, long double learning_rate, long double* training_inputs, long double* training_outputs) {
    network_output(network, training_inputs);
    backpropagate(network, training_outputs);
    train_network_step(network, learning_rate);
}

void train_network_on_multiple_io(Network* network, long double learning_rate, size_t io_len, long double** training_inputs, long double** training_outputs) {
    for(size_t i = 0; i < io_len; ++i) {
        train_network_on_single_io(network, learning_rate, training_inputs[i], training_outputs[i]);
    }
}

void train_network(Network* network, size_t epochs, long double learning_rate, size_t io_len, long double** training_inputs, long double** training_outputs) {
    for(size_t i = 0; i < epochs; ++i) {
        train_network_on_multiple_io(network, learning_rate, io_len, training_inputs, training_outputs);
    }
}