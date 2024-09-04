ALL: xor mnist

jpnn.o: jpnn.h jpnn.cc
	clang++ -Wall -Wextra -Os -c -g -o jpnn.o jpnn.cc

xor: xor.cc jpnn.o
	clang++ -Wall -Wextra -Os -g -o xor xor.cc jpnn.o

mnist: mnist.cc jpnn.o
	clang++ -Wall -Wextra -Os -g -o mnist mnist.cc jpnn.o

clean:
	rm -f jpnn.o xor mnist
