
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
#include <pthread.h>
#include <list>
#include <sched.h>
#include <sys/time.h>

#include <immintrin.h>
#include <mutex>

#define _GNU_SOURCE

using namespace std;

typedef struct {
	int jobType; 
	// 1->4: corners;	
	// 5->8:edges; 
	// 9->12:edge residues; 
	// 13:middle blocks; 
	// 14:fill last row; 
	// 15: fill last col; 
	// 16:last square
	int i;
	int j;
} job;

bool workerInitialized = false;
pthread_t* worker_threads;
int threadCt = -1;
int timeLayerSkip = -1;
int worker_width;
int worker_height;

typedef struct {
	list<job> jobList;
	int numOfAwaitingJobs;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	pthread_barrier_t barrier;

	int heightThreadLimit;
	int widthThreadLimit;
	float* input;
	float* output;
	float* conduct;
	int stepsGoal;
	int blockSize;
	int barrierCt;

} ThreadStatus;
ThreadStatus worker_status;

typedef struct {
	int tid;
}ThreadArgs;
ThreadArgs* worker_thread_args;

#define idx(i,j) ((i)*worker_width+(j))
#define idx_3d(i,j,t) ((t)*worker_width*worker_height+(i)*worker_width+(j))
#define min(x,y) (((x)>(y))?(y):(x))
#define max(x,y) (((x)>(y))?(x):(y))
#define abs(x) (((x)>0)?(x):(-(x)))



/* 
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
*/

void printMatrix(float* curr, int iStart, int jStart, int w, int h)
{
	cout<<"\t\t";
	for (int j=jStart;j<jStart+w;j++)
		cout<<j<<"\t";
	cout<<endl<<endl;
	for (int i=iStart;i<iStart+h;i++)
	{
		cout<<i<<"\t\t";
		for(int j=jStart;j<jStart+w;j++)
			cout<<(float)((int)(curr[idx(i,j)]*100))/100<<"\t";
		cout<<endl;
	}
	cout<<endl;
}

bool isEquals(float* temp, float* temp2, int width, int height) {
	for (int i = 0; i < height;i++)
	{
		for (int j=0;j<width;j++)
		{
			if (temp[idx(i,j)] != temp2[idx(i,j)])
				return false;
		}
	}
	return true;
}

inline void process_8(float* temp, float* temp2, float* conduct, int i, int j)
{
	// cout<<"process_8"<<endl;
	// cout<<isEquals(worker_status.input,temp,worker_width,worker_height)<<endl;
	//deal with (i,j) -> (i,j+7)
	__builtin_prefetch(temp+idx(i,j), 1, 3);
	__builtin_prefetch(temp2+idx(i,j+1), 1, 3);
	__builtin_prefetch(conduct+idx(i,j+1), 0, 1);

	__m256 k = _mm256_set1_ps(0.2);

	__m256 currTemp = _mm256_loadu_ps(temp+idx(i,j));

	__m256 leftTemp = _mm256_loadu_ps(temp+idx(i,j-1));
	__m256 rightTemp = _mm256_loadu_ps(temp+idx(i,j+1));
	__m256 leftCond = _mm256_loadu_ps(conduct+idx(i,j-1));
	__m256 rightCond = _mm256_loadu_ps(conduct+idx(i,j+1));
	__m256 leftRslt = _mm256_mul_ps(_mm256_sub_ps(leftTemp,currTemp), leftCond);
	__m256 rightRslt = _mm256_mul_ps(_mm256_sub_ps(rightTemp,currTemp), rightCond);
	// cout<<"avx:\t";
	// for (int jj=j;jj<j+9;jj++)
	// 	cout<<i<<","<<jj<<"="<<temp[idx(i-1,jj)]<<"\t";
	// cout<<endl;
	__m256 upTemp = _mm256_loadu_ps(temp+idx(i-1,j));
	// if (i==2022 && j==1)
	// 	cout<<"process_8 checkpoint 1"<<endl;
	__m256 downTemp = _mm256_loadu_ps(temp+idx(i+1,j));	
	// if (i==2022 && j==1)
	// 	cout<<"process_8 checkpoint 2"<<endl;
	__m256 upCond = _mm256_loadu_ps(conduct+idx(i-1,j));	
	// if (i==2022 && j==1)
	// 	cout<<"process_8 checkpoint 3"<<endl;
	__m256 downCond = _mm256_loadu_ps(conduct+idx(i+1,j));	
	// if (i==2022 && j==1)
	// 	cout<<"process_8 checkpoint 4"<<endl;
	__m256 upRslt = _mm256_mul_ps(_mm256_sub_ps(upTemp,currTemp), upCond);
	// if (i==2022 && j==1)
	// 	cout<<"process_8 checkpoint 5"<<endl;
	__m256 downRslt = _mm256_mul_ps(_mm256_sub_ps(downTemp,currTemp), downCond);
	// if (i==2022 && j==1)
	// 	cout<<"process_8 checkpoint 6"<<endl;

	__m256 currRslt = _mm256_add_ps(_mm256_add_ps(leftRslt,rightRslt),_mm256_add_ps(upRslt,downRslt));
	__m256 finalRslt = _mm256_fmadd_ps(currRslt,k,currTemp);
	_mm256_storeu_ps(temp2+idx(i,j), finalRslt);
	// if (i==2022 && j==1)
		// cout<<"process_8 checkpoint 7"<<endl;
}

inline void process_1(float* temp, float* temp2, float* conduct, int i, int j)
{
	//deal with (i,j) only
	temp2[idx(i,j)] = temp[idx(i,j)] +
					((temp[idx(i-1,j)] 
					- temp[idx(i,j)]) *conduct[idx(i-1,j)]
					+ (temp[idx(i+1,j)] 
					- temp[idx(i,j)]) *conduct[idx(i+1,j)]
					+ (temp[idx(i,j-1)]
					- temp[idx(i,j)]) *conduct[idx(i,j-1)]
					+ (temp[idx(i,j+1)] 
					- temp[idx(i,j)]) *conduct[idx(i,j+1)]
				)*0.2;
}

/*
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
// void horiProcess(float* temp, float* temp2, float* conduct, int threads, int substeps)
// {
	
// 		// cout<<"first line's first blocks"<<endl;
// 	//first line
// 		//first block of first line
// 			//upper left corner
// 	temp[idx(0,0)] = temp[idx(0,1)];
// 	for (int i=1;i<blockSize;i++)
// 	{
// 		//first col
// 		temp[idx(i,0)] = temp[idx(i,1)];

// 		for (int j=1;j<blockSize;j++)
// 		{
// 			//b0 other rows
// 			temp2[idx(i,j)] = temp[idx(i,j)] +
// 			(
//                         (temp[idx(i,j-1)]
// 						- temp[idx(i,j)]) *conduct[idx(i,j-1)]
						
// 						+ (temp[idx(i,j+1)] 
// 						- temp[idx(i,j)]) *conduct[idx(i,j+1)]
// 			)*0.2;
// 		}
// 	}
// 	// cout<<"first line's last blocks"<<endl;
// 		//last block of first line
// 			//upper right corner
// 	temp[idx(0,width-1)] = temp[idx(0,width-2)];
// 	for (int i=1;i<blockSize;i++)
// 	{
// 		//last col
// 		temp[idx(i,width-1)] = temp[idx(i,width-2)];

// 		for (int j=width-blockSize;j<width-1;j++)
// 		{
// 			//bm other rows
// 			temp2[idx(i,j)] = temp[idx(i,j)] +
// 			(
//                         (temp[idx(i,j-1)]
// 						- temp[idx(i,j)]) *conduct[idx(i,j-1)]
// 						+ (temp[idx(i,j+1)] 
// 						- temp[idx(i,j)]) *conduct[idx(i,j+1)]
// 			)*0.2;
// 		}
// 	}
// 		//middle blocks of first line
// 		// cout<<"first line's middle blocks"<<endl;
// 	for (int blockJ=blockSize; blockJ<width-blockSize;blockJ+=blockSize)
// 	{
// 		for (int i=1;i<blockSize;i++)
// 		{
// 			for (int j=blockJ;j<blockJ+blockSize;j++)
// 			{
// 				temp2[idx(i,j)] = temp[idx(i,j)] +
// 				(
// 							(temp[idx(i,j-1)]
// 							- temp[idx(i,j)]) *conduct[idx(i,j-1)]
// 							+ (temp[idx(i,j+1)] 
// 							- temp[idx(i,j)]) *conduct[idx(i,j+1)]
// 				)*0.2;
// 			}
// 		}
// 	}
// 	// cout<<"last line's fisrt blocks"<<endl;
// 	//last line
// 		//first block of last line
// 			//left down corner
// 	temp[idx(height-1,0)] = temp[idx(height-1,1)];

// 	for (int i=height-blockSize;i<height-1;i++)
// 	{
// 		//first col
// 		temp[idx(i,0)] = temp[idx(i,1)];

// 		for (int j=1;j<blockSize;j++)
// 		{
// 			temp2[idx(i,j)] = temp[idx(i,j)] +
// 			(
//                         (temp[idx(i,j-1)]
// 						- temp[idx(i,j)]) *conduct[idx(i,j-1)]
						
// 						+ (temp[idx(i,j+1)] 
// 						- temp[idx(i,j)]) *conduct[idx(i,j+1)]
// 			)*0.2;
// 		}
// 	}
// 	// cout<<"last line's last blocks"<<endl;
// 		//last block of last line
// 			//right down corner
// 	temp[idx(height-1,width-1)] = temp[idx(height-1,width-2)];

// 	for (int i=height-blockSize;i<height-1;i++)
// 	{
// 		//last col
// 		temp[idx(i,width-1)] = temp[idx(i,width-2)];

// 		for (int j=width-blockSize;j<width-1;j++)
// 		{
// 			temp2[idx(i,j)] = temp[idx(i,j)] +
// 			(
//                         (temp[idx(i,j-1)]
// 						- temp[idx(i,j)]) *conduct[idx(i,j-1)]
// 						+ (temp[idx(i,j+1)] 
// 						- temp[idx(i,j)]) *conduct[idx(i,j+1)]
// 			)*0.2;
// 		}
// 	}
// 	// cout<<"last line's middle blocks"<<endl;
// 		//middle blocks of last line
// 	for (int blockJ=blockSize; blockJ<width-blockSize;blockJ+=blockSize)
// 	{
// 		for (int i=height-blockSize;i<height-1;i++)
// 		{
// 			for (int j=blockJ;j<blockJ+blockSize;j++)
// 			{
// 				temp2[idx(i,j)] = temp[idx(i,j)] +
// 				(
// 							(temp[idx(i,j-1)]
// 							- temp[idx(i,j)]) *conduct[idx(i,j-1)]
// 							+ (temp[idx(i,j+1)] 
// 							- temp[idx(i,j)]) *conduct[idx(i,j+1)]
// 				)*0.2;
// 			}
// 		}
// 	}
// 	//middle lines
// 	for (int blockI=blockSize; blockI<=height-2*blockSize; blockI+=blockSize)
// 	{
// 		// cout<<"first block of this middle rows"<<endl;
// 		//first block of last line
// 		for (int i=blockI;i<blockI+blockSize;i++)
// 		{
// 			//first col
// 			temp[idx(i,0)] = temp[idx(i,1)];

// 			for (int j=1;j<blockSize;j++)
// 			{
// 				temp2[idx(i,j)] = temp[idx(i,j)] +
// 				(
// 							(temp[idx(i,j-1)]
// 							- temp[idx(i,j)]) *conduct[idx(i,j-1)]
							
// 							+ (temp[idx(i,j+1)] 
// 							- temp[idx(i,j)]) *conduct[idx(i,j+1)]
// 				)*0.2;
// 			}
// 		}
// 		//cout<<"last block of this middle rows"<<endl;
// 		//last block of last line
// 		for (int i=blockI;i<blockI+blockSize;i++)
// 		{
// 			//last col
// 			temp[idx(i,width-1)] = temp[idx(i,width-2)];

// 			for (int j=width-blockSize;j<width-1;j++)
// 			{
// 				temp2[idx(i,j)] = temp[idx(i,j)] +
// 				(
// 							(temp[idx(i,j-1)]
// 							- temp[idx(i,j)]) *conduct[idx(i,j-1)]
							
// 							+ (temp[idx(i,j+1)] 
// 							- temp[idx(i,j)]) *conduct[idx(i,j+1)]
// 				)*0.2;
// 			}
// 		}
// 		//cout<<"middle block of this middle rows"<<endl;
// 		//middle blocks of last line
		
// 		for (int blockJ=blockSize; blockJ<=width-2*blockSize;blockJ+=blockSize)
// 		{
// 			for (int i=blockI;i<blockI+blockSize;i++)
// 			{
// 				for (int j=blockJ;j<blockJ+blockSize;j++)
// 				{
// 					temp2[idx(i,j)] = temp[idx(i,j)] +
// 					(
// 								(temp[idx(i,j-1)]
// 								- temp[idx(i,j)]) *conduct[idx(i,j-1)]
								
// 								+ (temp[idx(i,j+1)] 
// 								- temp[idx(i,j)]) *conduct[idx(i,j+1)]
// 					)*0.2;
// 				}
// 			}
// 		}
// 	}
// }
// }
*/


// may/23 new content:

inline void leftDownCorner()
{
	//double checked on 5/26, no edges
	//ii = h-bs
	//jj = 0 
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	float* output=worker_status.output + timeLayerSkip *(stepsGoal-1);
	
	//first step
	// 	//edges
	// for ( int i = worker_height-2-blockSize; i <= worker_height-2; i++ )
	// 	worker_status.input[idx(i,0)] = worker_status.input[idx(i,1)];
	// for ( int j = 0; j < 1+blockSize; j++ )
	// 	worker_status.input[idx(worker_height-1,j)] = worker_status.input[idx(worker_height-2,j)];
		//rest
	for ( int i = worker_height-1-blockSize; i <= worker_height-2; i++ )
	{
		for ( int j = 1; j < 1+blockSize; j+=8 ) 
		{
			// cout<<"naive:\t";
			// for (int jjj=j;jjj<j+8;jjj++)
			// 	cout<<i-1<<" * "<<worker_width<<" + "<<j<<" = "<<idx(i-1,jjj)<<endl;
			// cout<<endl;
			process_8(worker_status.input, output , worker_status.conduct,i,j);
		}
	}
	// cout<<"leftDownCorner checkpoint 2"<<endl;
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		int iEnd = worker_height-blockSize+time-1;
		int jEnd = 1+blockSize-time;
		// 	//edges
		// for ( int i = worker_height-1; i>iEnd; i-- )
		// 	worker_status.input[idx(i,0)] = worker_status.input[idx(i,1)];
		// for ( int j = 0; j < jEnd; j++ )
		// 	worker_status.input[idx(worker_height-1,j)] = worker_status.input[idx(worker_height-2,j)];
			//rest
		for ( int i = worker_height -2; i>=iEnd; i-- ) 
		{
			int j=1;
			for (;j<jEnd-7;j+=8) 
			{
				// cout<<"time= "<<time<<"\tproc 8: "<<i<<"/"<<worker_height<<"\t"<<j<<"/"<<worker_width<<endl;
				process_8(input, output, worker_status.conduct,i,j);
			}
			for (;j<jEnd;j++) 
			{
				// cout<<"time= "<<time<<"\tproc 1: "<<i<<"/"<<worker_height<<"\t"<<j<<"/"<<worker_width<<endl;
				process_1(input, output, worker_status.conduct,i,j);
			}
		}
	}
	// cout<<"leftDownCorner checkpoint 3"<<endl;
}

inline void rightDownCorner()
{
	//double checked on 5/26, no edges
	//ii = h-bs
	//jj = w-bs
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	float* output=worker_status.output + timeLayerSkip *(stepsGoal-1);
	//first step
	for ( int i = worker_height-1-blockSize; i <= worker_height-2; i++ )  
	{
		for ( int j = worker_width-1-blockSize; j <worker_width-1; j+=8 ) 
			process_8(worker_status.input, output , worker_status.conduct,i,j);
	}
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		int iEnd = worker_height-1-blockSize+time;
		int jEnd = worker_width-1;
		for ( int i = worker_height-2; i>=iEnd; i-- ) 
		{
			int j=worker_width-1-blockSize+time;
			for (;j<jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void LeftUpCorner()
{
	//double checked on 5/26, no edges
	//ii = 0
	//jj = 0
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	float* output=worker_status.output + timeLayerSkip *(stepsGoal-1);
	//first step
	for ( int i = 1; i < 1+blockSize; i++ ) 
	{
		for ( int j = 1; j <1+blockSize; j+=8 ) 
			process_8(worker_status.input, output , worker_status.conduct,i,j);
	}
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		int iEnd = 1+blockSize-time;
		int jEnd = 1+blockSize-time;
		for ( int i = 1; i<iEnd; i++ ) 
		{
			int j=1;
			for (;j<jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void RightUpCorner()
{
	// ii = 0
	// jj = w-bs
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	float* output=worker_status.output + timeLayerSkip *(stepsGoal-1);
	//first step
	for ( int i = 1; i < 1+blockSize; i++ ) 
	{
		for ( int j = worker_width-1-blockSize; j <worker_width-1; j+=8 )
			process_8(worker_status.input, output , worker_status.conduct,i,j);
	}
	
	// cout<<"checkpoint 1"<<endl;
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		int iEnd = 1+blockSize-time;
		int jEnd = worker_width-1;
		for ( int i = 1; i<iEnd; i++ ) 
		{
			int j=worker_width-1-blockSize+time;
			for (;j<jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void leftEdge(int ii)
{
	//double checked on 5/26, no edges
	// blockSize <= ii <= height-2*blockSize
	// jj == 0
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	float* output=worker_status.output + timeLayerSkip *(stepsGoal-1);
	//first step
	for ( int i = ii; i < ii+blockSize; i++ ) 
	{
		for ( int j = 1; j <1+blockSize; j+=8 ) 
			process_8(worker_status.input, output , worker_status.conduct,i,j);
	}
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		int jEnd = 1+blockSize-time;
		for ( int i = ii-time; i<ii+blockSize-time; i++ ) 
		{
			int j=1;
			for (;j<jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void leftEdgeResidue()
{
	//double checked on 5/26, no edges
	// ii = height-blockSize
	// jj = 0
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		int iStart=worker_height-blockSize-1-time;
		int iEnd=worker_height-blockSize-2+time;
		for (int i=iStart; i<=iEnd; i++ ) 
		{
			int j=1;
			int jEnd=blockSize-time;
			for (;j<=jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<=jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void UpEdge(int jj)
{
	//double checked on 5/26, no edges
	// ii == 0
	// blockSize <= jj <= width-2*blockSize
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	float* output=worker_status.output + timeLayerSkip *(stepsGoal-1);
	//first step
	for ( int i = 1; i < 1+blockSize; i++ ) 
	{
		for ( int j = jj; j <jj+blockSize; j+=8 ) 
			process_8(worker_status.input, output , worker_status.conduct,i,j);
	}
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		int jEnd = jj+blockSize-time;
		for ( int i = 1; i<1+blockSize-time; i++ ) 
		{
			int j=jj-time;
			for (;j<jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void upEdgeResidue()
{
	//double checked on 5/26, no edges
	// ii = 0
	// jj = w-blockSize
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		for (int i=1; i<1+blockSize-time; i++ ) 
		{
			int j=worker_width-blockSize-1-time;
			int jEnd=worker_width-blockSize-2+time;
			for (;j<=jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<=jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void downEdge(int jj)
{
	//double checked on 5/26, no edges
	// ii == height-blockSize
	// blockSize <= jj <= width-2*blockSize
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	float* output=worker_status.output + timeLayerSkip *(stepsGoal-1);
	//first step
	for ( int i = worker_height-2; i >= worker_height-1-blockSize; i-- ) 
	{
		for ( int j = jj; j <jj+blockSize; j+=8 ) 
			process_8(worker_status.input, output , worker_status.conduct,i,j);
	}
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		int jEnd = jj+blockSize-time;
		for ( int i = worker_height-2; i >= worker_height-1-blockSize+time; i-- ) 
		{
			int j=jj-time;
			for (;j<jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void downEdgeResidue()
{
	//double checked on 5/26, no edges
	// ii = h
	// jj = w
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		for (int i=worker_height-2; i>worker_height-2-blockSize+time; i-- ) 
		{
			int j=worker_width-blockSize-1-time;
			int jEnd=worker_width-blockSize-2+time;
			for (;j<=jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<=jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void rightEdge(int ii)
{
	//double checked on 5/26, no edges
	// blockSize <= ii <= height-2*blockSize
	// jj == w-blockSize
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	float* output=worker_status.output + timeLayerSkip *(stepsGoal-1);
	//first step
	for ( int i = ii; i < ii+blockSize; i++ ) 
	{
		for ( int j = worker_width-blockSize-1; j <worker_width-1; j+=8 ) 
			process_8(worker_status.input, output , worker_status.conduct,i,j);
	}
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		int jEnd = worker_width-1;
		for ( int i = ii+time; i<ii+blockSize+time; i++ ) 
		{
			int j=worker_width-blockSize-1+time;
			for (;j<jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void rightEdgeResidue()
{
	//double checked on 5/26, no edges
	// ii = -1
	// jj = -1
	// ii = height-blockSize
	// jj = 0
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		int i=blockSize+1-time;
		int iEnd=blockSize-1+time;
		for (; i<=iEnd; i++ ) 
		{
			int j=worker_width-blockSize+time-1;
			int jEnd=worker_width-1;
			for (;j<jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void middleBlocks(int ii, int jj, int currStepsStart, int currStepsGoal, int currblockSize) 
{
	//double checked on 5/26, no edges
	//blockSize <= ii <= height-2*blockSize
	//blockSize <= jj <= width-2*blockSize
	int totalStepsGoal=worker_status.stepsGoal;
	int currStepsEnd = currStepsStart+currStepsGoal;
	if ( currStepsStart == 0 ) {
		// first step
		int iEnd=ii+worker_status.blockSize;
		float* output = worker_status.output + timeLayerSkip*(totalStepsGoal-1);
		for ( int i = ii; i < iEnd; i++ ) 
		{
			for ( int j = jj; j <jj+currblockSize; j+=8 ) 
				process_8(worker_status.input, output , worker_status.conduct,i,j);
		}
		currStepsStart = 1;
	}
	// next steps
	for (int time = currStepsStart; time < currStepsEnd; time++ ) 
	{
		float* input = worker_status.output + timeLayerSkip*(totalStepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(totalStepsGoal-time-1);
		//shift to left and up
		int iStart = ii - time;
		int jStart = jj - time;
		int iEnd=ii+worker_status.blockSize;
		int jEnd=jStart+currblockSize;
		for (int i=iStart;i<iEnd;i++) 
		{
			for (int j=jStart;j<jEnd;j+=8)
				process_8(input, output , worker_status.conduct,i,j);
		}
		
	}
}

// void middleBlocksHelper(int ii, int jj, int currStepsStart, int currNumOfSteps, int blockSize) 
// {
// 	//double checked on 5/26, no edges
// 	//keep dividing blocks until bS<=16 and steps<=16.
// 	if ( blockSize > 16 || currNumOfSteps > 16 ) {
// 		if ( blockSize > currNumOfSteps ) {
// 			int nextBlockSize = blockSize/2;
// 			middleBlocksHelper(ii, jj, currStepsStart, currNumOfSteps, nextBlockSize);
// 			middleBlocksHelper(ii, jj+nextBlockSize, currStepsStart, currNumOfSteps, nextBlockSize);
// 		} else {
// 			int nextNumOfSteps = currNumOfSteps/2;
// 			middleBlocksHelper(ii, jj, currStepsStart, nextNumOfSteps, blockSize);
// 			middleBlocksHelper(ii, jj, currStepsStart+nextNumOfSteps, nextNumOfSteps, blockSize);
// 		} 
// 		return;
// 	}
// 	middleBlocks(ii, jj, currStepsStart, currNumOfSteps, blockSize);
// }

inline void lastColUpsideDown(int ii)
{
	//0 <= ii <= h-3*blockSize
	//jj = w
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	ii += blockSize;
	int jj=worker_width-blockSize;
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		for (int i=ii-time+1; i<ii+blockSize-time+1; i++ ) 
		{
			int j=jj-time;
			int jEnd=jj+time-1;
			for (;j<jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void lastRowUpsideDown(int jj)
{
	//double checked on 5/26, no edges
	//ii = h
	//0 <= jj <= w-3*blockSize
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	int ii=worker_height-blockSize;
	jj += blockSize;
	for (int time=1;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		for (int i=ii-time; i<ii+time-1; i++ ) 
		{
			int j=jj-time+1;
			int jEnd=jj-time+blockSize+1;
			for (;j<jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

inline void lastSqaure()
{
	//ii = h
	//jj = w;
	int stepsGoal=worker_status.stepsGoal;
	int blockSize=worker_status.blockSize;
	int ii=worker_height-blockSize;
	int jj=worker_width-blockSize;
	for (int time=2;time<stepsGoal;time++)
	{
		//set pointers for input and output
		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
		for (int i=ii-time+1; i<ii+time-1; i++ ) 
		{
			int j=jj-time+1;
			int jEnd=jj+time-1;
			for (;j<jEnd-7;j+=8) 
				process_8(input, output, worker_status.conduct,i,j);
			for (;j<jEnd;j++) 
				process_1(input, output, worker_status.conduct,i,j);
		}
	}
}

// void threadFunc_edgeBlocks(int ii, int jj)
// {
// 	int stepsGoal=worker_status.stepsGoal;
// 	int blockSize=worker_status.blockSize;
// 	int condition=0;
// 	// (ii,jj) is the upper left coordinate of block
// 	if ((ii==0 || jj==0) && (ii+jj!= worker_width && ii+jj!= worker_height))
// 	{
// 		// No dependencies for corner	
// 		// Edges only depend on corner
// 		condition=-1;
// 		int iStart=(ii==0?1:0)+ii;
// 		int jStart=(jj==0?1:0)+jj;
// 		float* output=worker_status.output + timeLayerSkip *(stepsGoal-1);
// 		//first step
// 		for ( int i = iStart; i < ii+blockSize; i++ ) 
// 		{
// 			//process 8 values at each time to optimize with avx.
// 			for ( int j = jStart; j < jj+blockSize; j+=8 ) 
// 				process_8(worker_status.input, output , worker_status.conduct,i,j);
// 		}
// 	}
// 	//shift to up or left.
// 	for (int time=1;time<stepsGoal;time++)
// 	{
// 		//set pointers for input and output
// 		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
// 		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
// 		//shift to up if possible
// 		int iStart = (ii==0?1:-time)+ii;
// 		//shift to left if possible
// 		int jStart = (jj==0?1:-time)+jj;
// 		int iEnd = min(worker_width-1, ii + blockSize + condition*time);
// 		int jEnd = min(worker_height-1, jj + blockSize + condition*time);
// 		for ( int i = iStart; i<iEnd; i++ ) 
// 		{
// 			int j = jStart;
// 			for (;j<jEnd-7;j+=8) 
// 				process_8(input, output, worker_status.conduct,i,j);
// 			for (;j<jEnd;j++) 
// 				process_1(input, output, worker_status.conduct,i,j);
// 		}
// 	}
// }

// inline void recursionHelper(int ii, int jj, int currStepsStart, int currNumOfSteps, int blockSize) 
// {
// 	int stepsGoal=worker_status.stepsGoal;
// 	float* output = worker_status.output + timeLayerSkip*(stepsGoal-1);
// 	int currStepsEnd = currStepsStart+currNumOfSteps;
// 	// first step
// 	if ( currStepsStart == 0 ) {
// 		int iEnd=ii+blockSize;
// 		int jEnd=jj+worker_status.blockSize;
// 		for (int i=ii;i<iEnd;i++) 
// 		{
// 			for ( int j = jj; j < jEnd; j += 8 ) 
// 				process_8(worker_status.input, output , worker_status.conduct,i,j);
// 		}
// 		currStepsStart = 1;
// 	}
// 	// next steps
// 	for (int time = currStepsStart; time < currStepsEnd; time++ ) 
// 	{
// 		float* input = worker_status.output + timeLayerSkip*(stepsGoal-time);
// 		float* output = worker_status.output + timeLayerSkip*(stepsGoal-time-1);
// 		int iStart = ii - time;
// 		int jStart = jj - time;
// 		int iEnd=iStart+blockSize;
// 		int jEnd=jStart+worker_status.blockSize;
// 		for (int i=iStart;i<iEnd;i++) 
// 		{
// 			for (int j=jStart;j<jEnd;j+=8)
// 				process_8(input, output , worker_status.conduct,i,j);
// 		}
// 	}
// }

// void threadFunc_middleBlocks(int ii, int jj, int currStepsStart, int currNumOfSteps, int blockSize) 
// {
// 	//keep dividing blocks until bS<=16 and steps<=16.
// 	if ( blockSize > 16 || currNumOfSteps > 16 ) {
// 		if ( blockSize > currNumOfSteps ) {
// 			int nextBlockSize = blockSize/2;
// 			threadFunc_middleBlocks(ii, jj, currStepsStart, currNumOfSteps, nextBlockSize);
// 			threadFunc_middleBlocks(ii, jj+nextBlockSize, currStepsStart, currNumOfSteps, nextBlockSize);
// 		} else {
// 			int nextNumOfSteps = currNumOfSteps/2;
// 			threadFunc_middleBlocks(ii, jj, currStepsStart, nextNumOfSteps, blockSize);
// 			threadFunc_middleBlocks(ii, jj, currStepsStart+nextNumOfSteps, nextNumOfSteps, blockSize);
// 		} 
// 		return;
// 	}
// 	recursionHelper(ii, jj, currStepsStart, currNumOfSteps, blockSize);
// }

void* worker_thread(void* arg_) 
{
	ThreadArgs* arg = (ThreadArgs*)arg_;
	int tid = arg->tid;
	while (true) 
	{
		pthread_mutex_lock(&worker_status.mutex);
		if ( worker_status.jobList.empty() ) 
		{
			// cout<<"Thread "<<tid<<" got empty job list. "<<endl;
			pthread_mutex_unlock(&worker_status.mutex);
			// let thread to sleep for a little bit time to let newly created thread in
			// otherwise, at least on my computer(11900k win10), 
			// concurrency conflicts will happen.
			Sleep(2*threadCt+10);
			// pthread_yield();
			// this_thread::yield();
			sched_yield();
			// cout<<"Thread "<<tid<<" error code = "<<sched_yield()<<endl;
			continue;
		}
		job currJob = worker_status.jobList.front();
		if ( currJob.jobType != -1 ) 
		{
			worker_status.jobList.pop_front();
			// cout<<"Thread "<<tid<<" Doing job: "<<currJob.jobType<<endl;
			worker_status.numOfAwaitingJobs--;
		} 
		else 
		{
			worker_status.barrierCt++;
			// cout<<"Thread "<<tid<<" got last job. "<<endl;
			// cout<<"Thread "<<tid<<":\t"<<worker_status.barrierCt<<"\t"<<threadCt<<endl;
			if ( worker_status.barrierCt >= threadCt ) 
			{
				worker_status.barrierCt = 0;
				worker_status.jobList.pop_front();
				if ( worker_status.numOfAwaitingJobs == 0 ) 
					pthread_cond_signal(&worker_status.cond);
			}
		}
		pthread_mutex_unlock(&worker_status.mutex);

		if ( currJob.jobType == -1 ) 
		{
			// cout<<"Thread "<<tid<<" barrier waiting"<<endl;
			//sched_yield();
			pthread_barrier_wait(&worker_status.barrier);
			continue;
		} 
		if (currJob.jobType <=8)
		{
			if (currJob.jobType <=4)
			{
				//1 -> 4
				switch(currJob.jobType) 
				{
					case 1:
						leftDownCorner();
						break;
					case 2:
						rightDownCorner();
						break;
					case 3:
						LeftUpCorner();
						break;
					case 4:
						RightUpCorner();
						break;
				}
			}
			else
			{
				switch(currJob.jobType) 
				{
					case 5:
						UpEdge(currJob.j);
						break;
					case 6:
						downEdge(currJob.j);
						break;
					case 7:
						leftEdge(currJob.i);
						break;
					case 8:
						rightEdge(currJob.i);
						break;
				}
			}
		}
		else
		{
			if (currJob.jobType <=12)
			{
				//9 -> 12
				switch(currJob.jobType) 
				{
					case 9:
						upEdgeResidue();
						break;
					case 10:
						downEdgeResidue();
						break;
					case 11:
						leftEdgeResidue();
						break;
					case 12:
						rightEdgeResidue();
						break;
				}
			}
			else
			{
				switch(currJob.jobType) 
				{
					case 13:
						middleBlocks(currJob.i,currJob.j,0,
											worker_status.stepsGoal,
											16);
						break;
					case 14:
						lastRowUpsideDown(currJob.j);
						break;
					case 15:
						lastColUpsideDown(currJob.i);
						break;
					case 16:
						lastSqaure();
						break;
				}
			}
		}
		
		// cout<<"Thread "<<tid<<" finished job: "<<currJob.jobType<<endl;
	}
	return NULL;
}

void trapezoid_blocking(float* temp, float* temp2, float* conduct, int w, int h, int threads, int substeps)
{
	int blockSize = min(256,max(w,h)/threads/2);
	// int blockSize = 16;
	//assume blockSize can be divided by 8
	if (blockSize % 16 != 0)
		cout <<"Performance warning: blockSize = "<<blockSize<<", cannot be divided by 16. This will cause segmentFault and earily terminate."<<endl;
	// else
	// 	cout<<"threads = "<<threads<<", blockSize = "<<blockSize<<endl;
	//assume substeps > 0, meaningless o.w.
	if (substeps & (substeps-1))
		cout <<"Performance warning: substeps = "<<substeps<<", is not a power of 2. This will cause performance issue."<<endl;
	//my step should = (bS-7)
	if (workerInitialized)
	{
		// check if thread count is changed.
		if (threadCt != threads)
			cout <<"Performance warning: threads = "<<threads<<", is changed. This will cause performance issue."<<endl;
	}
	else
	{
		//First step
		threadCt = threads;
		worker_threads = (pthread_t*)malloc(sizeof(pthread_t)*threads);
		worker_thread_args = (ThreadArgs*)malloc(sizeof(ThreadArgs*)*threads);
		for ( int i = 0; i < threads; i++ ) {
			worker_thread_args[i].tid = i;
			pthread_create(&worker_threads[i], NULL, worker_thread, &worker_thread_args[i]);
			// cout<<"Thread "<<worker_thread_args[i].tid<<" spawned."<<endl;
		}
		pthread_mutex_init(&worker_status.mutex,NULL);
		pthread_cond_init(&worker_status.cond,NULL);
		pthread_barrier_init(&worker_status.barrier, NULL, threads);
		workerInitialized = true;
	}
	pthread_mutex_lock(&worker_status.mutex);

	int hLimit = h/blockSize;
	int wLimit= w/blockSize;
	worker_status.heightThreadLimit = hLimit;
	worker_status.widthThreadLimit = wLimit;
	worker_status.input = temp;
	worker_status.conduct = conduct;
	worker_status.output = temp2;
	worker_status.blockSize = blockSize;
	worker_status.barrierCt = 0;
	worker_height = h;

	int jobType = 1;
	int jobCt = 0;

	// cout<<isEquals(worker_status.input,temp,worker_width,worker_height)<<endl;
	//No dependencies for corner
	for (int i=0;i<4;i++)
	{
		job w;
		w.jobType=jobType++;
		worker_status.jobList.push_back(w);
		jobCt++;
	}
	worker_status.numOfAwaitingJobs+=jobCt;
	jobCt=0;

	//Edges only depend on corner
		//up
	for (int jj=blockSize; jj <= w-2*blockSize;jj+=blockSize)
	{
		job w;
		w.jobType=jobType;
		w.j=jj;
		worker_status.jobList.push_back(w);
		jobCt++;
	}
	jobType++;
	worker_status.numOfAwaitingJobs+=jobCt;
	jobCt=0;
		//down
	for (int jj=blockSize; jj <= w-2*blockSize;jj+=blockSize)
	{
		job w;
		w.jobType=jobType;
		w.j=jj;
		worker_status.jobList.push_back(w);
		jobCt++;
	}
	jobType++;
	worker_status.numOfAwaitingJobs+=jobCt;
	jobCt=0;
		//left
	for (int ii=blockSize; ii <= h-2*blockSize;ii+=blockSize)
	{
		job w;
		w.jobType=jobType;
		w.i=ii;
		worker_status.jobList.push_back(w);
		jobCt++;
	}
	jobType++;
	worker_status.numOfAwaitingJobs+=jobCt;
	jobCt=0;
		//right
	for (int ii=h-2*blockSize; ii >= blockSize;ii-=blockSize)
	{
		job w;
		w.jobType=jobType;
		w.i=ii;
		worker_status.jobList.push_back(w);
		jobCt++;
	}
	jobType++;
	worker_status.numOfAwaitingJobs+=jobCt;
	jobCt=0;
		//4 residuals
	for (int i=0;i<4;i++)
	{
		job w;
		w.jobType=jobType++;
		worker_status.jobList.push_back(w);
		jobCt++;
	}
	worker_status.numOfAwaitingJobs+=jobCt;
	jobCt=0;
	// middle blocks
	for (int ii=blockSize; ii <= h-2*blockSize;ii+=blockSize)
	{
		for (int jj=blockSize; jj <= w-2*blockSize;jj+=16)
		{
			job w;
			w.jobType=jobType;
			w.i=ii;
			w.j=jj;
			worker_status.jobList.push_back(w);
			jobCt++;
		}
		worker_status.numOfAwaitingJobs+=jobCt;
		jobCt=0;
	}
	jobType++;
	// fill last row
	for (int jj=0; jj <= w-3*blockSize;jj+=blockSize)
	{
		job w;
		w.jobType=jobType;
		w.j=jj;
		worker_status.jobList.push_back(w);
		jobCt++;
	}
	jobType++;
	// fill last col
	for (int ii=0; ii <= h-3*blockSize;ii+=blockSize)
	{
		job w;
		w.jobType=jobType;
		w.i=ii;
		worker_status.jobList.push_back(w);
		jobCt++;
	}
	// fill last square
	job wLast;
	wLast.jobType=++jobType;
	worker_status.jobList.push_back(wLast);
	jobCt++;
	// last job lets goooooooooooooooooooooooooooooo
	job w_end;
	w_end.jobType=-1;
	worker_status.jobList.push_back(w_end);
	
	worker_status.numOfAwaitingJobs += jobCt;
	
	// if (!workerInitialized)
	// {
	// 	//First step
	// 	threadCt = threads;
	// 	worker_threads = (pthread_t*)malloc(sizeof(pthread_t)*threads);
	// 	worker_thread_args = (ThreadArgs*)malloc(sizeof(ThreadArgs*)*threads);
	// 	for ( int i = 0; i < threads; i++ ) {
	// 		worker_thread_args[i].tid = i;
	// 		pthread_create(&worker_threads[i], NULL, worker_thread, &worker_thread_args[i]);
	// 		cout<<"Thread "<<worker_thread_args[i].tid<<" spawned."<<endl;
	// 	}
	// 	pthread_mutex_init(&worker_status.mutex,NULL);
	// 	pthread_cond_init(&worker_status.cond,NULL);
	// 	pthread_barrier_init(&worker_status.barrier, NULL, threads);
	// 	workerInitialized = true;
	// }
	// pthread_mutex_lock(&worker_status.mutex);
	while (worker_status.numOfAwaitingJobs > 0) 
	{
		pthread_cond_wait(&worker_status.cond, &worker_status.mutex);
		// cout<<"worker_status.numOfAwaitingJobs = "<<worker_status.numOfAwaitingJobs<<endl;
	}
	// cout<<"worker_status.numOfAwaitingJobs = "<<worker_status.numOfAwaitingJobs<<endl;
	pthread_mutex_unlock(&worker_status.mutex);
}





void step_optimized(float* temp, float* temp2, float* conduct, int w, int h, int threads, int substeps)
{
	// threads=1;
	// substeps=16;
	worker_width = w;
	timeLayerSkip = w*h;
	worker_status.stepsGoal = substeps;
	// worker_width = w;
	// for (int i=0;i<h;i++)
	// {
	// 	for (int j=0;j<w;j++)
	// 	{
	// 		if (temp[idx(i,j)] != 0)
	// 			cout<<i<<","<<j<<"="<<temp[idx(i,j)]<<"\t";
	// 	}
	// }

	// _aligned_free(temp2);
	// temp2 = (float*)_aligned_malloc(sizeof(float)*w*h*substeps, 64);
	// int ii=256-15;
	// int jj=256-15;
	trapezoid_blocking(temp, temp, conduct, w, h, threads, substeps);
	// printMatrix(temp,ii,jj,30,30);

	// int ii=256-15;
	// int jj=256-15;
	// cout<<"time = 0"<<endl;
	// printMatrix(temp,ii,jj,30,30);
	// for (int time=1; time<substeps;time++)
	// {
	// 	cout<<"time ="<<time<<endl;
	// 	printMatrix(temp2+ w*h*(substeps-time),ii,jj,30,30);
	// }

	/*
	//first approach: process horizontal lines and the transpose and then process horizontal lines again to avoid wasting reading any cache lines.
	float* transpose_conduct = (float*)_aligned_malloc(sizeof(float)*width*height, 64);
	transpose_block_SSE4x4(conduct, transpose_conduct, height, width);
	for ( int i = 0; i <substeps; i+=2) 
	{
			horiProcess(temp, temp2, conduct, threads, substeps);
			
			float* transpose_temp = (float*)_aligned_malloc(sizeof(float)*width*height, 64);
			float* transpose_temp2 = (float*)_aligned_malloc(sizeof(float)*width*height, 64);
			transpose_block_SSE4x4(temp, transpose_temp, height, width);
			transpose_block_SSE4x4(temp2, transpose_temp2, height, width);
			
			horiProcess_again(transpose_temp, transpose_temp2, transpose_conduct, threads, substeps);

			horiProcess(transpose_temp2, transpose_temp, transpose_conduct, threads, substeps);

			transpose_block_SSE4x4(transpose_temp2, temp2, height, width);
			transpose_block_SSE4x4(transpose_temp, temp, height, width);

			
			horiProcess_again(temp2, temp, conduct, threads, substeps);
			
			_aligned_free(transpose_temp);
			_aligned_free(transpose_temp2);
	
	*/
}
