all:
	g++ -o obj/stencil main.cpp stencil.cpp -Wall -pedantic -pthread -O3 -mavx -mavx2 -mfma -g -fopenmp 
	gcc -o obj/datagen datagen.cpp -Wall -pedantic

clean:
	rm -rf obj
