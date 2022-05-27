#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>

#include <immintrin.h>

#include <list>
#include <mutex>

#define MIN(x,y) (((x)>(y))?(y):(x))
#define MAX(x,y) (((x)>(y))?(x):(y))
#define ABS(x) (((x)>0)?(x):(-(x)))

typedef struct {
	bool fence;
	int x;
	int y;
} WorkItem;


bool g_initialized = false;
pthread_t* g_threads;
int threadcnt = -1;
int stepelems = -1;
int g_width;
int g_height;

#define IDX(x,y) (((y)*g_width)+x)

typedef struct {
	std::list<WorkItem> list;
	int left;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	pthread_barrier_t barrier;

	int hcnt;
	int wcnt;
	float* from;
	float* to;
	float* conduct;
	int steps;
	int blockwidth;
	int barriercnt;

} ThreadStatus;
ThreadStatus g_status;

typedef struct {
	int tid;
}ThreadArgs;
ThreadArgs* g_thread_args;

inline void stencil(int i, int j, float* from, float* to, float* conduct) {
	to[IDX(i,j)] = 
		from[IDX(i,j)] +
		(
		 (from[IDX(i-1,j)] 
		  - from[IDX(i,j)]) *conduct[IDX(i-1,j)]

		 + (from[IDX(i+1,j)] 
			 - from[IDX(i,j)]) *conduct[IDX(i+1,j)]


		 + (from[IDX(i,j-1)]
			 - from[IDX(i,j)]) *conduct[IDX(i,j-1)]
		 + (from[IDX(i,j+1)] 
			 - from[IDX(i,j)]) *conduct[IDX(i,j+1)]

		)*0.2;
}

inline void stencil_avx(int i, int j, float* from, float* to, float* conduct) {
	__builtin_prefetch(to+IDX(i,j), 1, 3);
	//__builtin_prefetch(from+IDX(i,j+1), 0, 2);
	//__builtin_prefetch(conduct+IDX(i,j+1), 0, 1);

	__m256 fv = _mm256_loadu_ps(from+IDX(i,j));
	__m256 fc = _mm256_set1_ps(0.2);

	__m256 fvb = _mm256_loadu_ps(from+IDX(i-1,j));
	__m256 fvn = _mm256_loadu_ps(from+IDX(i+1,j));
	__m256 cvb = _mm256_loadu_ps(conduct+IDX(i-1,j));
	__m256 cvn = _mm256_loadu_ps(conduct+IDX(i+1,j));

	__m256 ls = _mm256_mul_ps(_mm256_sub_ps(fvb,fv), cvb);
	__m256 rs = _mm256_mul_ps(_mm256_sub_ps(fvn,fv), cvn);
	
	__m256 cvd = _mm256_loadu_ps(conduct+IDX(i,j+1));
	__m256 fvd = _mm256_loadu_ps(from+IDX(i,j+1));
	__m256 fvu = _mm256_loadu_ps(from+IDX(i,j-1));
	__m256 cvu = _mm256_loadu_ps(conduct+IDX(i,j-1));


	__m256 us = _mm256_mul_ps(_mm256_sub_ps(fvu,fv), cvu);
	__m256 ds = _mm256_mul_ps(_mm256_sub_ps(fvd,fv), cvd);

	__m256 ss = _mm256_add_ps(_mm256_add_ps(ls,rs),_mm256_add_ps(us,ds));

	__m256 tt = _mm256_fmadd_ps(ss,fc,fv);

	_mm256_storeu_ps(to+IDX(i,j), tt);

}

void proc_init_corner_naive(bool xd, bool yd, int xo, int yo) {
	//for (int i = 0; i < g_status.steps
	int steps = g_status.steps;
	int bw = g_status.blockwidth;
	float* ft = g_status.to + stepelems*(steps-1);

	int xs = (xd?1:0)+xo;
	int ys = (yd?1:0)+yo;
	for ( int j = ys; j < yo+bw; j++ ) {
		for ( int i = xs; i < xo+bw; i += 8 ) {
			stencil_avx(i,j,g_status.from, ft, g_status.conduct);
		}
	}
	for ( int t = 1; t < steps; t++ ) {
		float* ft = g_status.to + stepelems*(steps-1-t);
		float* ff = g_status.to + stepelems*(steps-t);
		xs = (xd?1:-t)+xo;
		ys = (yd?1:-t)+yo;
		int xl = xo + bw-t;
		int yl = yo + bw-t;
		for ( int j = ys; j < yl; j++ ) {
			//for ( int i = xs; i < xl; i++ ) {
			int i = xs;
			for ( ; i < xl-7; i += 8 ) {
				stencil_avx(i,j,ff,ft, g_status.conduct);
			}
			for ( ; i < xl; i++ ) {
				stencil(i,j,ff,ft, g_status.conduct);
			}
		}
	}
}

void proc_end_corner_naive(bool xd, bool yd, int xo, int yo) {
	int steps = g_status.steps;
	int bw = g_status.blockwidth;
	for ( int t = 1; t < steps; t++ ) {
		float* ft = g_status.to + stepelems*(steps-1-t);
		float* ff = g_status.to + stepelems*(steps-t);
		
		int xs = xo - t;
		int ys = yo - t;
		int xl = MIN(g_width-1, xo + bw);
		int yl = MIN(g_height-1, yo + bw);

		if ( xl - xs >= 8 ) {
			for ( int j = ys; j < yl; j++ ) {
				for ( int i = xs; i < xl; i+= 8 ) {
					stencil_avx(i,j,ff,ft, g_status.conduct);
				}
			}
		} else {
			for ( int j = ys; j < yl; j++ ) {
				for ( int i = xs; i < xl; i++ ) {
					stencil(i,j,ff,ft, g_status.conduct);
				}
			}
		}

	}
}

inline void proc_mid_avx(int xo, int yo, int to, int w, int tt) {
	int steps = w;
	float* ft = g_status.to + stepelems*(g_status.steps-1);
	//printf( "%d %d %d -- %d %d\n", xo, yo, to, w, tt );
	
	int tg = to+tt;
	if ( to == 0 ) {
		for ( int j = yo; j < yo+steps; j ++ ) {
			//for ( int i = xo; i < xo+steps; i += 8 ) {
			for ( int i = xo; i < xo+g_status.blockwidth; i += 8 ) {
				stencil_avx(i,j,g_status.from, ft, g_status.conduct);
				//if ( g_status.from[IDX(i,j)] > 0 ) printf( "%f!!\n", ft[IDX(i,j)] );
				//printf( "%d %d -- %d -- %f!!\n", i, j, IDX(i,j), ft[IDX(i,j)] );
			}
		}
		to = 1;
	}
	
	for ( int t = to; t < tg; t++ ) {
		float* ft = g_status.to + stepelems*(g_status.steps-1-t);
		float* ff = g_status.to + stepelems*(g_status.steps-t);
		int xs = xo - t;
		int ys = yo - t;
		//int xl = xs + steps;
		int xl = xs + g_status.blockwidth;
		int yl = ys + steps;
		//printf( "%d %d %d %d\n", xs, ys, xl, yl );
		for ( int j = ys; j < yl; j ++ ) {
			for ( int i = xs; i < xl; i += 8 ) {
				stencil_avx(i,j,ff,ft, g_status.conduct);
				//if ( ft[IDX(i,j)] > 0 )
			}
		}
	}
}


void proc_mid_re(int xo, int yo, int to, int w, int tt) {
	if ( w > 16 || tt > 16 ) {
		if ( w > tt ) {
			int nw = w/2;
			proc_mid_re(xo, yo, to, nw, tt);
			//proc_mid_re(xo+nw, yo, to, nw, tt);
			
			proc_mid_re(xo, yo+nw, to, nw, tt);
			//proc_mid_re(xo+nw, yo+nw, to, nw, tt);
			return;
		} else {
			int nt = tt/2;
			proc_mid_re(xo, yo, to, w, nt);
			proc_mid_re(xo, yo, to+nt, w, nt);
			return;
		} 
	}
	
	//proc_mid_single(xo,yo,to,w,tt);
	proc_mid_avx(xo,yo,to,w,tt);
	
}
void* worker_thread(void* arg_) {
	ThreadArgs* arg = (ThreadArgs*)arg_;
	int tid = arg->tid;
	int proccnt = 0;

	while (true) {
		pthread_mutex_lock(&g_status.mutex);
		if ( g_status.list.empty() ) {
			pthread_mutex_unlock(&g_status.mutex);
			pthread_yield();
			continue;
		}
		WorkItem w = g_status.list.front();
		if ( !w.fence ) {
			g_status.list.pop_front();
			g_status.left--;
		} else {
			g_status.barriercnt++;
			if ( g_status.barriercnt >= threadcnt ) {
				g_status.barriercnt = 0;
				g_status.list.pop_front();
				
				if ( g_status.left == 0 ) pthread_cond_signal(&g_status.cond); //only used here to signal parent thread
				//printf( "Signalling cond %d\n", tid );
				//fflush(stdout);
			}
		}
		pthread_mutex_unlock(&g_status.mutex);

		if ( w.fence ) {
			pthread_barrier_wait(&g_status.barrier);
			//printf( "Thread %d -- proc %d\n", tid, proccnt );
			continue;
		} 

		proccnt++;
		//printf( "Thread %d -- Processing %d %d => %d\n", tid, w.x, w.y, g_status.left );



		if ( w.x == g_status.wcnt || w.y == g_status.hcnt ) { // end corner
			proc_end_corner_naive(w.x==g_status.wcnt-1, w.y==g_status.hcnt-1, w.x*g_status.blockwidth, w.y*g_status.blockwidth);
		} else if ( w.x == 0 || w.y == 0 ) { //init corner
			proc_init_corner_naive(w.x==0, w.y==0, w.x*g_status.blockwidth, w.y*g_status.blockwidth);
		} else {
			//proc_mid_naive(w.x*g_status.blockwidth, w.y*g_status.blockwidth);
			//proc_mid_naive_avx(w.x*g_status.blockwidth, w.y*g_status.blockwidth);
			proc_mid_re(w.x*g_status.blockwidth, w.y*g_status.blockwidth, 0, g_status.blockwidth, g_status.steps);
		}
	}
	return NULL;
}

void
step_optimized(float* temp, float* temp2, float* conduct, int width, int height, int threads, int substeps) {
	if ( (width & ( width-1 )) || ( height & ( height-1) )) {
		printf( "WARNING: width/height mod substeps not zero\n" );
		return;
	}
	if ( substeps & ( substeps - 1 ) ) {
		printf( "WARNING: substeps not power of two\n" );
		return;
	}

	if (g_initialized) {
		if ( threadcnt != threads ) {
			printf( "WARNING: thread count changed\n" );
			return;
		}
	} else {
		threadcnt = threads;
		g_threads = (pthread_t*)malloc(sizeof(pthread_t)*threads);
		//g_status.lists = (std::list<WorkItem>**)malloc(sizeof(std::list<WorkItem>*)*threads);
		g_thread_args = (ThreadArgs*)malloc(sizeof(ThreadArgs*)*threads);

		for ( int i = 0; i < threads; i++ ) {
			//g_status.lists[i] = new std::list<WorkItem>();
			g_thread_args[i].tid = i;

			printf("Thread spawn\n");
			pthread_create(&g_threads[i], NULL, worker_thread, &g_thread_args[i]);
		}
		pthread_mutex_init(&g_status.mutex,NULL);
		pthread_cond_init(&g_status.cond,NULL);
		pthread_barrier_init(&g_status.barrier, NULL, threads);
		g_initialized = true;
	}

	int bw = MIN(256,MAX(height,width)/threads/2); // heuristic...
	//int bw = 128;
	printf( "bw: %d\n", bw );
	int hcnt = height/bw +1;
	int wcnt = width/bw +1;
	int target = 0;
	pthread_mutex_lock(&g_status.mutex);
	g_status.hcnt = hcnt-1;
	g_status.wcnt = wcnt-1;
	g_status.from = temp;
	g_status.conduct = conduct;
	g_status.to = temp2;
	g_status.steps = substeps;
	g_status.blockwidth = bw;
	g_status.barriercnt = 0;
	stepelems = width*height;
	g_width = width;
	g_height = height;

	int enqcount = 0;
	for ( int i = 0; i < hcnt+wcnt; i++ ) { //diagonals, of hcnt+1 x wcnt+1 matrix
		int cs = MAX(0, i-hcnt);
		int cnt = MIN(i, MIN(wcnt-cs, hcnt));
			
		int lj = cnt-1;
		int r = MIN(hcnt,i)-lj-1;
		int c = cs+lj;
		WorkItem w;
		w.x = r; w.y = c; w.fence = false;
		g_status.list.push_back(w);
		enqcount++;

		for ( int j = 0; j < cnt-1; j++ ) {
			int r = MIN(hcnt,i)-j-1;
			int c = cs+j;
			WorkItem w;
			w.x = r; w.y = c; w.fence = false;
			g_status.list.push_back(w);

			//printf( "%d: %d %d -> %d %ld\n", i, r,c, target, g_status.lists[target]->size() );
			enqcount++;
			target = (target+1)%threads;
		}

		w.x = -1; w.y = -1; w.fence = true;
		//g_status.lists[j]->push_back(w);
		g_status.list.push_back(w);
	}
	g_status.left += enqcount;

	while (g_status.left > 0) {
		/*
		struct timespec ts;
		struct timeval now;
		gettimeofday(&now, NULL);
		ts.tv_sec = now.tv_sec;
		ts.tv_nsec = (now.tv_usec+10000UL)*1000UL;
		//pthread_cond_timedwait(&g_status.cond, &g_status.mutex, &ts);
		*/
		pthread_cond_wait(&g_status.cond, &g_status.mutex);
		//printf( "%d \n", g_status.left );
	}
	pthread_mutex_unlock(&g_status.mutex);

}

