// Version 0.1: 2020-04-15

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <iostream>
#include <Windows.h>

#include <ctime>
#include <chrono>

#define SUBSTEP 32

inline int idx(int i, int j, int width) {
	return i*width+j;
}

void printMatrix(float* curr, int iStart, int jStart, int w, int h, int width)
{
	for (int i=iStart;i<iStart+h;i++)
	{
		for(int j=jStart;j<jStart+w;j++)
			std::cout<<curr[idx(i,j,width)]<<"\t";
		std::cout<<std::endl;
	}
	std::cout<<std::endl;
}

inline void transpose4x4_SSE(float *A, float *B, const int lda, const int ldb) {
    __m128 row1 = _mm_load_ps(&A[0*lda]);
    __m128 row2 = _mm_load_ps(&A[1*lda]);
    __m128 row3 = _mm_load_ps(&A[2*lda]);
    __m128 row4 = _mm_load_ps(&A[3*lda]);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_store_ps(&B[0*ldb], row1);
     _mm_store_ps(&B[1*ldb], row2);
     _mm_store_ps(&B[2*ldb], row3);
     _mm_store_ps(&B[3*ldb], row4);
}

inline void transpose_block_SSE4x4(float *A, float *B, const int h, const int w) {
    #pragma omp parallel for
    for(int i=0; i<h; i+=64) {
        for(int j=0; j<w; j+=64) {
            //int max_i2 = i+block_size < n ? i + block_size : n;
            //int max_j2 = j+block_size < m ? j + block_size : m;
            for(int i2=i; i2< i+64; i2+=4) {
                for(int j2=j; j2< j+64; j2+=4) {
                    transpose4x4_SSE(&A[i2*w +j2], &B[j2*h + i2], w, h);
                }
            }
        }
    }
}

void 
step_naive(float* temp, float* temp2, float* conduct, int width, int height) {
	for ( int i = 0; i < height; i++ ) {
		temp[idx(0,i,width)] = temp[idx(1,i,width)];
		temp[idx(height-1, i,width)] = temp[idx(height-2,i,width)];
	}
	for ( int i = 0; i < width; i++ ) {
		temp[idx(i,0,width)] = temp[idx(i,1,width)];
		temp[idx(i,width-1,width)] = temp[idx(i,width-2,width)];
	}
	for ( int i = 1; i < width-1; i++ ) {
		for ( int j = 1; j < height-1; j++ ) {
			temp2[idx(i,j,width)] = 
				temp[idx(i,j,width)] +
				(
					(temp[idx(i-1,j,width)] 
					- temp[idx(i,j,width)]) *conduct[idx(i-1,j,width)]

					+ (temp[idx(i+1,j,width)] 
					- temp[idx(i,j,width)]) *conduct[idx(i+1,j,width)]


					+ (temp[idx(i,j-1,width)]
					- temp[idx(i,j,width)]) *conduct[idx(i,j-1,width)]
					+ (temp[idx(i,j+1,width)] 
					- temp[idx(i,j,width)]) *conduct[idx(i,j+1,width)]

				)*0.2;
		}
	}
	return;
}

void step_naive_edited(float* temp, float* temp2, float* conduct, int width, int height) 
{
	for ( int i = 0; i < height; i++ ) {
		temp[idx(0,i,width)] = temp[idx(1,i,width)];
		temp[idx(height-1, i,width)] = temp[idx(height-2,i,width)];
	}
	for ( int i = 1; i < width-1; i++ ) {
		for ( int j = 1; j < height-1; j++ ) {
			temp2[idx(i,j,width)] = 
				temp[idx(i,j,width)] +
				(
					(temp[idx(i-1,j,width)] 
					- temp[idx(i,j,width)]) *conduct[idx(i-1,j,width)]

					+ (temp[idx(i+1,j,width)] 
					- temp[idx(i,j,width)]) *conduct[idx(i+1,j,width)]

				)*0.2;
		}
	}
	
	float* transpose_temp = (float*)_aligned_malloc(sizeof(float)*width*height, 64);
	float* transpose_temp2 = (float*)_aligned_malloc(sizeof(float)*width*height, 64);
	float* transpose_conduct = (float*)_aligned_malloc(sizeof(float)*width*height, 64);
	transpose_block_SSE4x4(conduct, transpose_conduct, height, width);
	transpose_block_SSE4x4(temp, transpose_temp, height, width);
	transpose_block_SSE4x4(temp2, transpose_temp2, height, width);

	for ( int i = 0; i < height; i++ ) {
		transpose_temp[idx(0,i,height)] = transpose_temp[idx(1,i,height)];
		transpose_temp[idx(height-1, i,height)] = transpose_temp[idx(height-2,i,height)];
	}
	for ( int i = 1; i < height-1; i++ ) {
		for ( int j = 1; j < height-1; j++ ) {
			transpose_temp2[idx(j,i,height)] += 
				(
					(transpose_temp[idx(j-1,i,height)] 
					- transpose_temp[idx(j,i,height)]) *transpose_conduct[idx(j-1,i,height)]

					+ (transpose_temp[idx(j+1,i,height)] 
					- transpose_temp[idx(j,i,height)]) *transpose_conduct[idx(j+1,i,height)]

				)*0.2;
		}
	}
	
	
	memset( temp2, 0 ,sizeof( temp2));
	transpose_block_SSE4x4(transpose_temp2, temp2, height, width);
	transpose_block_SSE4x4(transpose_temp, temp, height, width);
	_aligned_free(transpose_temp);
	_aligned_free(transpose_temp2);
	_aligned_free(transpose_conduct);
	return;
}

extern void step_optimized(float* temp, float* temp2, float* conduct, int width, int height, int threads, int substeps);

void step(float* temp, float* temp2, float* conduct, int width, int height, int threads, int substeps, bool naive) 
{
	if ( naive ) {
		for ( int i = 0; i <substeps; i++ ) {
			step_naive(temp, temp2, conduct, width, height);
			if ( i < substeps - 1) {
				float* t = temp;
				temp = temp2;
				temp2 = t;
			}
		}
	} else step_optimized(temp, temp2, conduct, width, height, threads, substeps);
}



int main (int argc, char** argv) {
	printf( "usage: ./%s [steps] [threads] [initial state file] [naive?]\n", argv[0] );
	if ( SUBSTEP%2 ) {
		printf( "ERROR: SUBSTEP should be power of 2!\n" );
		exit(1);
	}


	int steps = 256;
	if ( argc >= 2 ) steps = atoi(argv[1]);
	
	int threads = 1;
	if ( argc >= 3 ) threads = atoi(argv[2]);
	
	const char* filename = "init.dat";
	if ( argc >= 4 ) {
		filename = argv[3];
	}

	bool isnaive = true;
	if ( argc >= 5 ) {
		if ( argv[4][0] == 'N' || argv[4][0] == 'n' ) isnaive = false;
	}

	FILE* fin = fopen(filename, "rb");

	if ( !fin ) {
		printf( "Input file %s not found!\n", filename );
		exit(1);
	}



	int width;
	int height;
	if ( !fread(&width, sizeof(int), 1, fin) || !fread(&height, sizeof(int), 1, fin) ) {
		printf( "Input file read failed\n" );
		exit(1);
	}
	printf( "Reading data file width, height = %d %d\n", width, height );
	float* temp = (float*)_aligned_malloc(sizeof(float)*width*height*SUBSTEP,32);
	float* temp2 = (float*)_aligned_malloc(sizeof(float)*width*height*SUBSTEP,32);
	float* conduct = (float*)_aligned_malloc(sizeof(float)*width*height,32);

	if ( fread(temp, sizeof(float), width*height, fin) != (size_t)(width*height)
		|| fread(conduct, sizeof(float), width*height, fin) != (size_t)(width*height) ) {

		printf( "Input file read failed\n" );
		exit(1);
	}
	fclose(fin);


	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point now;
	std::chrono::microseconds duration_micro;

	printf( "Starting %s stencil calculation for %d steps\n", isnaive?"naive":"optimized", steps );
	start = std::chrono::high_resolution_clock::now();

	int ii = 0;
	// printMatrix(temp,256-8,256-8,16,16,width);
	for (int i=0;i<2048;i++)
	{
		for (int j=0;j<2048;j++)
			temp[idx(i,j,width)]=500;
	}
	while (ii < steps) {
		step(temp, temp2, conduct, width, height, threads, SUBSTEP, isnaive);
		// step(temp, temp2, conduct, width, height, threads, 1, isnaive);
		// break;
		ii += SUBSTEP;
		printf( "\t Step %8d\n", ii );
	}

	now = std::chrono::high_resolution_clock::now();
	duration_micro = std::chrono::duration_cast<std::chrono::microseconds> (now-start);
	printf( "Done : %f s\n", 0.000001f*duration_micro.count() );
	printf( "%s stencil calculation done\n", isnaive?"naive":"optimized" );

	FILE* fout = fopen("output.dat", "wb");
	fwrite(&width, sizeof(int), 1, fout);
	fwrite(&height, sizeof(int), 1, fout);
	fwrite(temp, sizeof(float), width*height, fout);
	fwrite(conduct, sizeof(float), width*height, fout);
	fclose(fout);

	printf( "All done! Exiting...\n" );
}
