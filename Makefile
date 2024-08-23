all: build
	./run
build:
	gcc -ggdb -o run conv.c
