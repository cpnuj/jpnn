xor: xor.cc jpnn.h jpnn.cc
	clang++ -Wall -Wextra -Os -g -o xor xor.cc jpnn.cc

clean:
	rm -f xor
