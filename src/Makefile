main: main.o utils.o neuron.o layer.o network.o
	@gcc -o main main.o utils.o neuron.o layer.o network.o -lm

main.o: main.c
	@gcc main.c -o main.o -c

utils.o: utils/utils.c
	@gcc utils/utils.c -o utils.o -c

neuron.o: core/neuron.c
	@gcc core/neuron.c -o neuron.o -c

layer.o: core/layer.c
	@gcc core/layer.c -o layer.o -c

network.o: core/network.c
	@gcc core/network.c -o network.o -c

clean:
	@rm main.o utils.o neuron.o layer.o network.o

clean-all:
	@rm main
	@make clean

run: main
	@./main