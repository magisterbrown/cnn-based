all: build
	./run
build:
	gcc -fopenmp -ggdb -o run conv.c
