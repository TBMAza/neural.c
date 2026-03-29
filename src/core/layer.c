#include "layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void init_layer(Layer* layer, size_t neurons_no, size_t inputs_per_neuron) {
    if(neurons_no == 0) {
        neurons_no = 1;
    }
    layer->neurons_no = neurons_no;
    layer->outputs = NULL;
    layer->neurons = calloc(neurons_no, sizeof(Neuron));
    for(size_t i = 0; i < neurons_no; ++i) {
        init_neuron(layer->neurons + i, inputs_per_neuron);
    }
}
void print_layer(Layer* layer, char* indentation) {
    if(!(layer && layer->neurons)) {
        return;
    }
    if(!indentation) {
        indentation = "";
    }
    long double* placeholder = calloc(layer->neurons_no, sizeof(long double));
	if(!layer->outputs) {
		layer->outputs = placeholder;
	}
    printf("%s{\n", indentation);
    printf("%s\t.neurons_no: %zu\n", indentation, layer->neurons_no);
    printf("%s\t.outputs: {", indentation);
    for(size_t i = 0; i < layer->neurons_no; ++i) {
        printf("%Lf, ", layer->outputs[i]);
    }
    printf("}\n");
    printf("%s\t.neurons:\n", indentation);

    char next_indentation[strlen(indentation) + 2];
    strcpy(next_indentation, indentation);
    strcat(next_indentation, "\t");
    for(size_t i = 0; i < layer->neurons_no; ++i) {
        print_neuron(layer->neurons + i, next_indentation);
    }
    printf("%s}\n", indentation);
    if(layer->outputs == placeholder) {
		layer->outputs = NULL;
	}
	free(placeholder);
}
void layer_output(Layer* layer, long double* inputs) {
    if(!(layer && inputs)) {
		return;
	}
    long double* output = malloc(sizeof(long double) * layer->neurons_no);
    for(size_t i = 0; i < layer->neurons_no; ++i) {
        neuron_output(layer->neurons + i, inputs);
        output[i] = layer->neurons[i].output;
    }
    free(layer->outputs);
    layer->outputs = output;
}
void train_layer_step(Layer* layer, long double learning_rate) {
    if(!layer) {
        return;
    }
    for(size_t i = 0; i < layer->neurons_no; ++i) {
        train_neuron_step(layer->neurons + i, learning_rate);
    }
}
void free_layer(Layer* layer) {
    for(size_t i = 0; i < layer->neurons_no; ++i) {
        free_neuron(layer->neurons + i);
    }
    free(layer->neurons);
    free(layer->outputs);
}