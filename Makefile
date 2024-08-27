all: build
	./run
build:
	gcc -ggdb -o run main.c primitives/conv.c primitives/loss.c
