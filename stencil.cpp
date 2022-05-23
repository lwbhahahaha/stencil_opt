
//note: added [-std=c++17 -stdlib=libc++] to makefile for compiling on MacOS

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <iostream>
#include <Windows.h>

#include <ctime>
#include <chrono>
#include <thread>
#include <tchar.h>
#include <strsafe.h>
#define BUF_SIZE 255

using namespace std;

int blockSize;
int MAX_THREADS ;
int width;
int height;
inline int idx(int i, int j) {
	return i*width+j;
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
    for(int i=0; i<h; i+=blockSize) {
        for(int j=0; j<w; j+=blockSize) {
            //int max_i2 = i+block_size < n ? i + block_size : n;
            //int max_j2 = j+block_size < m ? j + block_size : m;
            for(int i2=i; i2< i+blockSize; i2+=4) {
                for(int j2=j; j2< j+blockSize; j2+=4) {
                    transpose4x4_SSE(&A[i2*w +j2], &B[j2*h + i2], w, h);
                }
            }
        }
    }
}

// struct thread_data
// {
//  int x;
//  int y;
//  float* temp;
// float* temp2; 
// float* conduct;
//  int width;
// int height; 
//  thread_data(int i, int j, float* t1, float* t2, float* c, int w, int h)
//  {
// 	x=i;
// 	y=j;
// 	temp=t1;
// 	temp2=t2;
// 	conduct=c;
// 	width=w;
// 	height=h;
//  }
// };

// DWORD WINAPI thread_func(LPVOID lpParameter)
// {
//  thread_data *td = (thread_data*)lpParameter;
//  block_cache_optimized(td->x, td->y, td->temp, td->temp2, td->conduct, td->width, td->height);
//  return 0;
// }
void process_block_avx(float* temp, float* temp2, float* conduct, int i, int j)
{
	__builtin_prefetch(temp2+idx(i,j), 1, 3);
	//__builtin_prefetch(temp+idx(i,j+1), 0, 2);
	//__builtin_prefetch(conduct+idx(i,j+1), 0, 1);

	__m256 k = _mm256_set1_ps(0.2);
	__m256 currTemp = _mm256_loadu_ps(temp+idx(i,j));
	__m256 leftTemp = _mm256_loadu_ps(temp+idx(i,j-1));
	__m256 rightTemp = _mm256_loadu_ps(temp+idx(i,j+1));
	__m256 leftCond = _mm256_loadu_ps(conduct+idx(i,j-1));
	__m256 rightCond = _mm256_loadu_ps(conduct+idx(i,j+1));
	__m256 leftRslt = _mm256_mul_ps(_mm256_sub_ps(leftTemp,currTemp), leftCond);
	__m256 rightRslt = _mm256_mul_ps(_mm256_sub_ps(rightTemp,currTemp), rightCond);
	__m256 currRslt = _mm256_add_ps(leftRslt,rightRslt);
	__m256 finalRslt = _mm256_add_ps(_mm256_fmadd_ps(currRslt,k,currTemp),_mm256_loadu_ps(temp2+idx(i,j)));
	_mm256_storeu_ps(temp2+idx(i,j), finalRslt);
}
void horiProcess_avx(float* temp, float* temp2, float* conduct, int threads, int substeps)
{
	// cout<<"\tfirst line's first blocks"<<endl;
	for (int j=0;j<blockSize;j++)
		_mm256_storeu_ps(temp+idx(0,j), _mm256_loadu_ps(temp+idx(1,j)));
	for (int i=1;i<blockSize;i++)
	{
		for (int j=1;j<blockSize;j++)
			process_block_avx(temp,temp2,conduct,i,j);
	}
	//bm
		// cout<<"\tfirst line's last blocks"<<endl;
	for (int j=width-blockSize;j<width-1;j++)
		_mm256_storeu_ps(temp+idx(0,j), _mm256_loadu_ps(temp+idx(1,j)));
	for (int i=1;i<blockSize;i++)
	{
		for (int j=width-blockSize;j<width-1;j++)
			process_block_avx(temp,temp2,conduct,i,j);
	}
	//b2 -> bm-1
		// cout<<"\tfirst line's middle blocks"<<endl;
	for (int blockJ=blockSize; blockJ<width-blockSize;blockJ+=blockSize)
	{
		for (int j=blockJ;j<blockJ+blockSize;j++)
			_mm256_storeu_ps(temp+idx(0,j), _mm256_loadu_ps(temp+idx(1,j)));
		for (int i=1;i<blockSize;i++)
		{
			for (int j=blockJ;j<blockJ+blockSize;j++)
				process_block_avx(temp,temp2,conduct,i,j);
		}
	}
	//last line
		// cout<<"\tlast line's fisrt blocks"<<endl;
	for (int j=1;j<blockSize;j++)
		_mm256_storeu_ps(temp+idx(height-1,j), _mm256_loadu_ps(temp+idx(height-2,j)));
	for (int i=height-blockSize;i<height-1;i++)
	{
		for (int j=1;j<blockSize;j++)
			process_block_avx(temp,temp2,conduct,i,j);
	}
		// cout<<"\tlast line's last blocks"<<endl;
	for (int j=width-blockSize;j<width;j++)
		_mm256_storeu_ps(temp+idx(height-1,j), _mm256_loadu_ps(temp+idx(height-2,j)));
	for (int i=height-blockSize;i<height-1;i++)
	{
		for (int j=width-blockSize;j<width-1;j++)
			process_block_avx(temp,temp2,conduct,i,j);
	}
	//last line's middle blocks
		// cout<<"\tlast line's middle blocks"<<endl;
	for (int blockJ=blockSize; blockJ<width-blockSize;blockJ+=blockSize)
	{
		for (int j=blockJ;j<blockJ+blockSize;j++)
			_mm256_storeu_ps(temp+idx(height-1,j), _mm256_loadu_ps(temp+idx(height-2,j)));
		for (int i=height-blockSize;i<height-1;i++)
		{
			for (int j=blockJ;j<blockJ+blockSize;j++)
				process_block_avx(temp,temp2,conduct,i,j);
		}
	}
	//middle rows
	// cout<<"\tmiddle rows"<<endl;
	for (int blockI=blockSize; blockI<height-blockSize; blockI+=blockSize)
	{
		//first block of this row
		// cout<<"\tfirst block of this middle rows"<<endl;
		for (int i=blockI;i<blockI+blockSize;i++)
		{
			for (int j=1;j<blockSize;j++)
				process_block_avx(temp,temp2,conduct,i,j);
		}
		//last block of this row
		// cout<<"\tlast block of this middle rows"<<endl;
		for (int i=blockI;i<blockI+blockSize;i++)
		{
			for (int j=width-blockSize;j<width-1;j++)
				process_block_avx(temp,temp2,conduct,i,j);
		}
		//middle blocks
		// cout<<"\tmiddle block of this middle rows"<<endl;
		for (int blockJ=blockSize; blockJ<width-2*blockSize;blockJ+=blockSize)
		{
			for (int i=blockI;i<blockI+blockSize;i++)
			{
				for (int j=blockJ;j<blockJ+blockSize;j++)
					process_block_avx(temp,temp2,conduct,i,j);
			}
		}
	}
}

void horiProcess_again(float* temp, float* temp2, float* conduct, int threads, int substeps)
{
	
		// cout<<"first line's first blocks"<<endl;
	//first line
		//first block of first line
			//upper left corner
	temp[idx(0,0)] = temp[idx(0,1)];
	for (int i=1;i<blockSize;i++)
	{
		//first col
		temp[idx(i,0)] = temp[idx(i,1)];

		for (int j=1;j<blockSize;j++)
		{
			//b0 other rows
			temp2[idx(i,j)] +=
			(
                        (temp[idx(i,j-1)]
						- temp[idx(i,j)]) *conduct[idx(i,j-1)]
						
						+ (temp[idx(i,j+1)] 
						- temp[idx(i,j)]) *conduct[idx(i,j+1)]
			)*0.2;
		}
	}
	// cout<<"first line's last blocks"<<endl;
		//last block of first line
			//upper right corner
	temp[idx(0,width-1)] = temp[idx(0,width-2)];
	for (int i=1;i<blockSize;i++)
	{
		//last col
		temp[idx(i,width-1)] = temp[idx(i,width-2)];

		for (int j=width-blockSize;j<width-1;j++)
		{
			//bm other rows
			temp2[idx(i,j)] +=
			(
                        (temp[idx(i,j-1)]
						- temp[idx(i,j)]) *conduct[idx(i,j-1)]
						+ (temp[idx(i,j+1)] 
						- temp[idx(i,j)]) *conduct[idx(i,j+1)]
			)*0.2;
		}
	}
		//middle blocks of first line
		// cout<<"first line's middle blocks"<<endl;
	for (int blockJ=blockSize; blockJ<width-blockSize;blockJ+=blockSize)
	{
		for (int i=1;i<blockSize;i++)
		{
			for (int j=blockJ;j<blockJ+blockSize;j++)
			{
				temp2[idx(i,j)] +=
				(
							(temp[idx(i,j-1)]
							- temp[idx(i,j)]) *conduct[idx(i,j-1)]
							+ (temp[idx(i,j+1)] 
							- temp[idx(i,j)]) *conduct[idx(i,j+1)]
				)*0.2;
			}
		}
	}
	// cout<<"last line's fisrt blocks"<<endl;
	//last line
		//first block of last line
			//left down corner
	temp[idx(height-1,0)] = temp[idx(height-1,1)];

	for (int i=height-blockSize;i<height-1;i++)
	{
		//first col
		temp[idx(i,0)] = temp[idx(i,1)];

		for (int j=1;j<blockSize;j++)
		{
			temp2[idx(i,j)] +=
			(
                        (temp[idx(i,j-1)]
						- temp[idx(i,j)]) *conduct[idx(i,j-1)]
						
						+ (temp[idx(i,j+1)] 
						- temp[idx(i,j)]) *conduct[idx(i,j+1)]
			)*0.2;
		}
	}
	// cout<<"last line's last blocks"<<endl;
		//last block of last line
			//right down corner
	temp[idx(height-1,width-1)] = temp[idx(height-1,width-2)];

	for (int i=height-blockSize;i<height-1;i++)
	{
		//last col
		temp[idx(i,width-1)] = temp[idx(i,width-2)];

		for (int j=width-blockSize;j<width-1;j++)
		{
			temp2[idx(i,j)] +=
			(
                        (temp[idx(i,j-1)]
						- temp[idx(i,j)]) *conduct[idx(i,j-1)]
						+ (temp[idx(i,j+1)] 
						- temp[idx(i,j)]) *conduct[idx(i,j+1)]
			)*0.2;
		}
	}
	// cout<<"last line's middle blocks"<<endl;
		//middle blocks of last line
	for (int blockJ=blockSize; blockJ<width-blockSize;blockJ+=blockSize)
	{
		for (int i=height-blockSize;i<height-1;i++)
		{
			for (int j=blockJ;j<blockJ+blockSize;j++)
			{
				temp2[idx(i,j)] +=
				(
							(temp[idx(i,j-1)]
							- temp[idx(i,j)]) *conduct[idx(i,j-1)]
							+ (temp[idx(i,j+1)] 
							- temp[idx(i,j)]) *conduct[idx(i,j+1)]
				)*0.2;
			}
		}
	}
	//middle lines
	for (int blockI=blockSize; blockI<=height-2*blockSize; blockI+=blockSize)
	{
		// cout<<"first block of this middle rows"<<endl;
		//first block of last line
		for (int i=blockI;i<blockI+blockSize;i++)
		{
			//first col
			temp[idx(i,0)] = temp[idx(i,1)];

			for (int j=1;j<blockSize;j++)
			{
				temp2[idx(i,j)] +=
				(
							(temp[idx(i,j-1)]
							- temp[idx(i,j)]) *conduct[idx(i,j-1)]
							
							+ (temp[idx(i,j+1)] 
							- temp[idx(i,j)]) *conduct[idx(i,j+1)]
				)*0.2;
			}
		}
		//cout<<"last block of this middle rows"<<endl;
		//last block of last line
		for (int i=blockI;i<blockI+blockSize;i++)
		{
			//last col
			temp[idx(i,width-1)] = temp[idx(i,width-2)];

			for (int j=width-blockSize;j<width-1;j++)
			{
				temp2[idx(i,j)] +=
				(
							(temp[idx(i,j-1)]
							- temp[idx(i,j)]) *conduct[idx(i,j-1)]
							
							+ (temp[idx(i,j+1)] 
							- temp[idx(i,j)]) *conduct[idx(i,j+1)]
				)*0.2;
			}
		}
		//cout<<"middle block of this middle rows"<<endl;
		//middle blocks of last line
		
		for (int blockJ=blockSize; blockJ<=width-2*blockSize;blockJ+=blockSize)
		{
			for (int i=blockI;i<blockI+blockSize;i++)
			{
				for (int j=blockJ;j<blockJ+blockSize;j++)
				{
					temp2[idx(i,j)] +=
					(
								(temp[idx(i,j-1)]
								- temp[idx(i,j)]) *conduct[idx(i,j-1)]
								
								+ (temp[idx(i,j+1)] 
								- temp[idx(i,j)]) *conduct[idx(i,j+1)]
					)*0.2;
				}
			}
		}
	}
}

void horiProcess(float* temp, float* temp2, float* conduct, int threads, int substeps)
{
	
		// cout<<"first line's first blocks"<<endl;
	//first line
		//first block of first line
			//upper left corner
	temp[idx(0,0)] = temp[idx(0,1)];
	for (int i=1;i<blockSize;i++)
	{
		//first col
		temp[idx(i,0)] = temp[idx(i,1)];

		for (int j=1;j<blockSize;j++)
		{
			//b0 other rows
			temp2[idx(i,j)] = temp[idx(i,j)] +
			(
                        (temp[idx(i,j-1)]
						- temp[idx(i,j)]) *conduct[idx(i,j-1)]
						
						+ (temp[idx(i,j+1)] 
						- temp[idx(i,j)]) *conduct[idx(i,j+1)]
			)*0.2;
		}
	}
	// cout<<"first line's last blocks"<<endl;
		//last block of first line
			//upper right corner
	temp[idx(0,width-1)] = temp[idx(0,width-2)];
	for (int i=1;i<blockSize;i++)
	{
		//last col
		temp[idx(i,width-1)] = temp[idx(i,width-2)];

		for (int j=width-blockSize;j<width-1;j++)
		{
			//bm other rows
			temp2[idx(i,j)] = temp[idx(i,j)] +
			(
                        (temp[idx(i,j-1)]
						- temp[idx(i,j)]) *conduct[idx(i,j-1)]
						+ (temp[idx(i,j+1)] 
						- temp[idx(i,j)]) *conduct[idx(i,j+1)]
			)*0.2;
		}
	}
		//middle blocks of first line
		// cout<<"first line's middle blocks"<<endl;
	for (int blockJ=blockSize; blockJ<width-blockSize;blockJ+=blockSize)
	{
		for (int i=1;i<blockSize;i++)
		{
			for (int j=blockJ;j<blockJ+blockSize;j++)
			{
				temp2[idx(i,j)] = temp[idx(i,j)] +
				(
							(temp[idx(i,j-1)]
							- temp[idx(i,j)]) *conduct[idx(i,j-1)]
							+ (temp[idx(i,j+1)] 
							- temp[idx(i,j)]) *conduct[idx(i,j+1)]
				)*0.2;
			}
		}
	}
	// cout<<"last line's fisrt blocks"<<endl;
	//last line
		//first block of last line
			//left down corner
	temp[idx(height-1,0)] = temp[idx(height-1,1)];

	for (int i=height-blockSize;i<height-1;i++)
	{
		//first col
		temp[idx(i,0)] = temp[idx(i,1)];

		for (int j=1;j<blockSize;j++)
		{
			temp2[idx(i,j)] = temp[idx(i,j)] +
			(
                        (temp[idx(i,j-1)]
						- temp[idx(i,j)]) *conduct[idx(i,j-1)]
						
						+ (temp[idx(i,j+1)] 
						- temp[idx(i,j)]) *conduct[idx(i,j+1)]
			)*0.2;
		}
	}
	// cout<<"last line's last blocks"<<endl;
		//last block of last line
			//right down corner
	temp[idx(height-1,width-1)] = temp[idx(height-1,width-2)];

	for (int i=height-blockSize;i<height-1;i++)
	{
		//last col
		temp[idx(i,width-1)] = temp[idx(i,width-2)];

		for (int j=width-blockSize;j<width-1;j++)
		{
			temp2[idx(i,j)] = temp[idx(i,j)] +
			(
                        (temp[idx(i,j-1)]
						- temp[idx(i,j)]) *conduct[idx(i,j-1)]
						+ (temp[idx(i,j+1)] 
						- temp[idx(i,j)]) *conduct[idx(i,j+1)]
			)*0.2;
		}
	}
	// cout<<"last line's middle blocks"<<endl;
		//middle blocks of last line
	for (int blockJ=blockSize; blockJ<width-blockSize;blockJ+=blockSize)
	{
		for (int i=height-blockSize;i<height-1;i++)
		{
			for (int j=blockJ;j<blockJ+blockSize;j++)
			{
				temp2[idx(i,j)] = temp[idx(i,j)] +
				(
							(temp[idx(i,j-1)]
							- temp[idx(i,j)]) *conduct[idx(i,j-1)]
							+ (temp[idx(i,j+1)] 
							- temp[idx(i,j)]) *conduct[idx(i,j+1)]
				)*0.2;
			}
		}
	}
	//middle lines
	for (int blockI=blockSize; blockI<=height-2*blockSize; blockI+=blockSize)
	{
		// cout<<"first block of this middle rows"<<endl;
		//first block of last line
		for (int i=blockI;i<blockI+blockSize;i++)
		{
			//first col
			temp[idx(i,0)] = temp[idx(i,1)];

			for (int j=1;j<blockSize;j++)
			{
				temp2[idx(i,j)] = temp[idx(i,j)] +
				(
							(temp[idx(i,j-1)]
							- temp[idx(i,j)]) *conduct[idx(i,j-1)]
							
							+ (temp[idx(i,j+1)] 
							- temp[idx(i,j)]) *conduct[idx(i,j+1)]
				)*0.2;
			}
		}
		//cout<<"last block of this middle rows"<<endl;
		//last block of last line
		for (int i=blockI;i<blockI+blockSize;i++)
		{
			//last col
			temp[idx(i,width-1)] = temp[idx(i,width-2)];

			for (int j=width-blockSize;j<width-1;j++)
			{
				temp2[idx(i,j)] = temp[idx(i,j)] +
				(
							(temp[idx(i,j-1)]
							- temp[idx(i,j)]) *conduct[idx(i,j-1)]
							
							+ (temp[idx(i,j+1)] 
							- temp[idx(i,j)]) *conduct[idx(i,j+1)]
				)*0.2;
			}
		}
		//cout<<"middle block of this middle rows"<<endl;
		//middle blocks of last line
		
		for (int blockJ=blockSize; blockJ<=width-2*blockSize;blockJ+=blockSize)
		{
			for (int i=blockI;i<blockI+blockSize;i++)
			{
				for (int j=blockJ;j<blockJ+blockSize;j++)
				{
					temp2[idx(i,j)] = temp[idx(i,j)] +
					(
								(temp[idx(i,j-1)]
								- temp[idx(i,j)]) *conduct[idx(i,j-1)]
								
								+ (temp[idx(i,j+1)] 
								- temp[idx(i,j)]) *conduct[idx(i,j+1)]
					)*0.2;
				}
			}
		}
	}
}
bool isChanged(float* temp, float* temp2, int width, int height) {
	for (int i = 0; i < height;i++)
	{
		for (int j=0;j<width;j++)
		{
			if (temp[idx(i,j)] != temp2[idx(i,j)])
				return true;
		}
	}
	return false;

}


void printMatrix(float* curr, int w, int h)
{
	for (int i=0;i<h;i++)
	{
		for(int j=0;j<w;j++)
			cout<<curr[i*w+j]<<"\t";
		cout<<endl;
	}
}

void step_optimized(float* temp, float* temp2, float* conduct, int w, int h, int threads, int substeps)
{
	// blockSize=4;
	// w=8;
	// h=8;
	// float* before = (float*)_aligned_malloc(sizeof(float)*w*h, 64);
	// float* after = (float*)_aligned_malloc(sizeof(float)*w*h, 64);
	// for (int i=0;i<h;i++)
	// {
	// 	for(int j=0;j<w;j++)
	// 		before[i*w+j]=i*w+j;
	// }
	// cout<<"before:"<<endl;
	// printMatrix(before,w,h);
	// transpose_block_SSE4x4(before, after, w, h);
	// cout<<"transposed:,w,h);
	// transpose_block_SSE4x4(after, before, w, h);
	// cout<<"transposed"<<endl;
	// printMatrix(before,w,h);
	// return;


	width=w;
	height=h;
	blockSize=min(256,max(height,width)/threads/2);


	float* transpose_conduct = (float*)_aligned_malloc(sizeof(float)*width*height, 64);
	transpose_block_SSE4x4(conduct, transpose_conduct, height, width);
	for ( int i = 0; i <substeps; i+=2) 
	{
			// cout<<"\t\ti = "<<i<<endl;
			horiProcess(temp, temp2, conduct, threads, substeps);
			// horiProcess_avx(temp, temp2, conduct, threads, substeps);
			// cout <<"1 hori success"<<endl;

			float* transpose_temp = (float*)_aligned_malloc(sizeof(float)*width*height, 64);
			float* transpose_temp2 = (float*)_aligned_malloc(sizeof(float)*width*height, 64);
			transpose_block_SSE4x4(temp, transpose_temp, height, width);
			transpose_block_SSE4x4(temp2, transpose_temp2, height, width);
			// cout <<"1 transpose success"<<endl;
			horiProcess_again(transpose_temp, transpose_temp2, transpose_conduct, threads, substeps);
			// horiProcess_avx(transpose_temp, transpose_temp2, transpose_conduct, threads, substeps);
			// cout <<"2 hori success"<<endl;
		//next step
			horiProcess(transpose_temp2, transpose_temp, transpose_conduct, threads, substeps);
			// horiProcess_avx(transpose_temp, transpose_temp2, transpose_conduct, threads, substeps);
			// cout <<"3 hori success"<<endl;
			transpose_block_SSE4x4(transpose_temp2, temp2, height, width);
			transpose_block_SSE4x4(transpose_temp, temp, height, width);
			// cout <<"2 hotransposeri success"<<endl;

			
			horiProcess_again(temp2, temp, conduct, threads, substeps);
			// horiProcess_avx(temp, temp2, conduct, threads, substeps);
			// cout <<"4 hori success"<<endl;
			
			_aligned_free(transpose_temp);
			_aligned_free(transpose_temp2);
			// cout<<"transpose freed"<<endl;

			// _aligned_free(temp2);
			// float* temp2 = (float*)_aligned_malloc(sizeof(float)*width*height*substeps,32);
			// cout<<"temp2 freed"<<endl;
	}
	float* t = temp;
	temp = temp2;
	temp2 = t;
}

// void step_optimized(float* temp, float* temp2, float* conduct, int width, int height, int threads, int substeps)
// {
// 	MAX_THREADS=threads;
// 	blockSize=min(256,max(height,width)/threads/2);
// 	int ct=0;
// 	thread myThread[threads] = {};
// 	for ( int s = 0; s <substeps; s++ )
// 	{
// 		for ( int i = 0; i < height; i+=blockSize)
// 		{
// 			for ( int j = 0; j < width; j+=blockSize)
// 			{
// 				myThread[ct%threads] = thread(block_cache_optimized, i, j, temp, temp2, conduct, width, height);
// 				// myThread[ct%threads] = thread(testing, s, ct%threads, temp, temp2, conduct, width, height);
// 				ct++;
// 				if (ct==threads)
// 				{
// 					for (int i=threads-1;i>=0;i--)
// 						myThread[i].join();
// 					ct=0;
// 					thread myThread[threads] = {};
// 				}
// 			}
// 		}
// 		if ( s < substeps - 1) 
// 		{
// 			float* t = temp;
// 			temp = temp2;
// 			temp2 = t;
// 		}
// 	}
// }

// void step_optimized(float* temp, float* temp2, float* conduct, int width, int height, int threads, int substeps)
// {
// 	blockSize=min(256,max(height,width)/threads/2);
// 	int ct=0;
// 	thread myThread[threads] = {};
// 	for ( int s = 0; s <substeps; s++ )
// 	{
// 		for ( int i = 0; i < height; i+=blockSize)
// 		{
// 			for ( int j = 0; j < width; j+=blockSize)
// 			{
// 				myThread[ct%threads] = thread(block_cache_optimized, i, j, temp, temp2, conduct, width, height);
// 				// myThread[ct%threads] = thread(testing, s, ct%threads, temp, temp2, conduct, width, height);
// 				ct++;
// 				if (ct==threads)
// 				{
// 					for (int i=threads-1;i>=0;i--)
// 						myThread[i].join();
// 					ct=0;
// 					thread myThread[threads] = {};
// 				}
// 			}
// 		}
// 		if ( s < substeps - 1) 
// 		{
// 			float* t = temp;
// 			temp = temp2;
// 			temp2 = t;
// 		}
// 	}
// }



// DWORD WINAPI MyThreadFunction( LPVOID lpParam );
// void ErrorHandler(LPTSTR lpszFunction);

// // Sample custom data structure for threads to use.
// // This is passed by void pointer so it can be any data type
// // that can be passed using a single void pointer (LPVOID).
// typedef struct MyData {
//     int x;
//  	int y;
//  	float* temp;
// 	float* temp2; 
// 	float* conduct;
//  	int width;
// 	int height; 
// } MYDATA, *PMYDATA;


// int _tmain()
// {
//     PMYDATA pDataArray[MAX_THREADS];
//     DWORD   dwThreadIdArray[MAX_THREADS];
//     HANDLE  hThreadArray[MAX_THREADS]; 

//     // Create MAX_THREADS worker threads.

//     for( int i=0; i<MAX_THREADS; i++ )
//     {
//         // Allocate memory for thread data.

//         pDataArray[i] = (PMYDATA) HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY,
//                 sizeof(MYDATA));

//         if( pDataArray[i] == NULL )
//         {
//            // If the array allocation fails, the system is out of memory
//            // so there is no point in trying to print an error message.
//            // Just terminate execution.
//             ExitProcess(2);
//         }

//         // Generate unique data for each thread to work with.

//         pDataArray[i]->val1 = i;
//         pDataArray[i]->val2 = i+100;

//         // Create the thread to begin execution on its own.

//         hThreadArray[i] = CreateThread( 
//             NULL,                   // default security attributes
//             0,                      // use default stack size  
//             MyThreadFunction,       // thread function name
//             pDataArray[i],          // argument to thread function 
//             0,                      // use default creation flags 
//             &dwThreadIdArray[i]);   // returns the thread identifier 


//         // Check the return value for success.
//         // If CreateThread fails, terminate execution. 
//         // This will automatically clean up threads and memory. 

//         if (hThreadArray[i] == NULL) 
//         {
//            ErrorHandler(TEXT("CreateThread"));
//            ExitProcess(3);
//         }
//     } // End of main thread creation loop.

//     // Wait until all threads have terminated.

//     WaitForMultipleObjects(MAX_THREADS, hThreadArray, TRUE, INFINITE);

//     // Close all thread handles and free memory allocations.

//     for(int i=0; i<MAX_THREADS; i++)
//     {
//         CloseHandle(hThreadArray[i]);
//         if(pDataArray[i] != NULL)
//         {
//             HeapFree(GetProcessHeap(), 0, pDataArray[i]);
//             pDataArray[i] = NULL;    // Ensure address is not reused.
//         }
//     }

//     return 0;
// }


// DWORD WINAPI MyThreadFunction( LPVOID lpParam ) 
// { 
//     HANDLE hStdout;
//     PMYDATA pDataArray;

//     TCHAR msgBuf[BUF_SIZE];
//     size_t cchStringSize;
//     DWORD dwChars;

//     // Make sure there is a console to receive output results. 

//     hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
//     if( hStdout == INVALID_HANDLE_VALUE )
//         return 1;

//     // Cast the parameter to the correct data type.
//     // The pointer is known to be valid because 
//     // it was checked for NULL before the thread was created.
 
//     pDataArray = (PMYDATA)lpParam;

//     // Print the parameter values using thread-safe functions.

//     StringCchPrintf(msgBuf, BUF_SIZE, TEXT("Parameters = %d, %d\n"), 
//         pDataArray->val1, pDataArray->val2); 
//     StringCchLength(msgBuf, BUF_SIZE, &cchStringSize);
//     WriteConsole(hStdout, msgBuf, (DWORD)cchStringSize, &dwChars, NULL);

//     return 0; 
// } 



// void ErrorHandler(LPTSTR lpszFunction) 
// { 
//     // Retrieve the system error message for the last-error code.

//     LPVOID lpMsgBuf;
//     LPVOID lpDisplayBuf;
//     DWORD dw = GetLastError(); 

//     FormatMessage(
//         FORMAT_MESSAGE_ALLOCATE_BUFFER | 
//         FORMAT_MESSAGE_FROM_SYSTEM |
//         FORMAT_MESSAGE_IGNORE_INSERTS,
//         NULL,
//         dw,
//         MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
//         (LPTSTR) &lpMsgBuf,
//         0, NULL );

//     // Display the error message.

//     lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT, 
//         (lstrlen((LPCTSTR) lpMsgBuf) + lstrlen((LPCTSTR) lpszFunction) + 40) * sizeof(TCHAR)); 
//     StringCchPrintf((LPTSTR)lpDisplayBuf, 
//         LocalSize(lpDisplayBuf) / sizeof(TCHAR),
//         TEXT("%s failed with error %d: %s"), 
//         lpszFunction, dw, lpMsgBuf); 
//     MessageBox(NULL, (LPCTSTR) lpDisplayBuf, TEXT("Error"), MB_OK); 

//     // Free error-handling buffer allocations.

//     LocalFree(lpMsgBuf);
//     LocalFree(lpDisplayBuf);
// }