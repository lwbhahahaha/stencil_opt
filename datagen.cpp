#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

void set_rect(float* a, int width, int x, int y, int w, int h, float v) {
	for ( int i = 0; i < w; i++ ) {
		for ( int j = 0; j < h; j++ ) {
			a[(y+j)*width+(x+i)] = v;
		}
	}
}

int
main( int argc, char** argv) {
	if ( argc < 3 ) {
		printf( "usage: ./%s width height\n", argv[0] );
		exit(1);
	}
	int width;
	int height;

	width = atoi(argv[1]);
	height = atoi(argv[2]);

	printf( "%d %d\n", width, height );

	float* temp = (float*)malloc(sizeof(float)*width*height);
	float* conduct = (float*)malloc(sizeof(float)*width*height);

	for ( int i = 0; i < width*height; i++ ) {
		temp[i] = 0;
		conduct[i] = 1.0;
	}

	printf( "Init data\n" );
	fflush(stdout);

	for ( int i = 0; i < 8; i++ ) {
		for ( int j = 0; j < 8; j++ ) {
			int xd = width/8;
			int yd = height/8;

			int xoff = xd*i;
			int yoff = yd*j;
			
			temp[yoff*width+xoff] = 500.0;
		}
	}
	set_rect(conduct, width, 0,0, width/2, height/2, 0.2);
	set_rect(conduct, width, 0,height*3/4, width, height/4, 0.5);
	set_rect(conduct, width, width/2 + 32,0, width/2-32, height/4, 0.2);


	printf( "Datagen!\n" );
	fflush(stdout);

	FILE* fout = fopen("init.dat", "wb");
	fwrite(&width, sizeof(int), 1, fout);
	fwrite(&height, sizeof(int), 1, fout);
	fwrite(temp, sizeof(float), width*height, fout);
	fwrite(conduct, sizeof(float), width*height, fout);
	fclose(fout);

	printf( "Done!\n" );

}
