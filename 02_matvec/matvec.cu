#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <cooperative_groups.h>
using namespace cooperative_groups;

// CUDA runtime
#include <cuda_runtime.h>

#define DTYPE float

const int maxThreadsPerBlock = 1024;

__device__ DTYPE mult(DTYPE *a, DTYPE *x, int size)
{
    int tid_x = threadIdx.x;

    int row = blockIdx.x;

    int bid_y = blockIdx.y;

    int bdim_x = blockDim.x;

    int col = tid_x + bid_y * bdim_x;

    return a[row * size + col] * x[col];
}

__device__ DTYPE reduce(thread_group g, DTYPE *cache, DTYPE val)
{
    int gtid_x = g.thread_rank();
    cache[gtid_x] = val;
    g.sync();
    for (int k = g.size() / 2; k > 0; k >>= 1)
    {
       
        if (gtid_x < k)
        {
            cache[gtid_x] += cache[gtid_x + k];
        }
        g.sync();
    }

    return cache[0];
}

__global__ void kernelAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    __shared__ DTYPE cache[maxThreadsPerBlock];

    DTYPE val = mult(a, x, size);

    thread_group g = this_thread_block();
    auto tileIdx = g.thread_rank() / 32;
    DTYPE* t = &cache[32 * tileIdx];

    thread_group tile = tiled_partition(g, 32);

    DTYPE sum = reduce(tile, t, val);

    //printf("%f \n", cache[0]);

    if (tile.thread_rank() == 0) atomicAdd(&y[blockIdx.x], sum);

    //if (row < size && col < size) atomicAdd(&y[row], a[row * size + col] * x[col]);
}



//A mit Werten füllen (hier einfach 1en)
void fillA(DTYPE *a, int size)
{
   for (int i=0;i<size*size;i++)
      a[i]=1.0;
}

//X mit Werten füllen 
void fillX(DTYPE *x, int size)
{
   for (int i=0;i<size;i++)
      x[i]= (DTYPE)(i+1);
}

void hostAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    //TODO: Hier soll der Host A*x=y berechnen
    for (unsigned int i = 0; i < size; i++)
    {
        y[i] = 0;
        for (unsigned int j = 0; j < size; j++)
        {
            y[i] += a[j * size + i] * x[j];
        }
    }
}




bool checkResult(DTYPE *yh, DTYPE *yd, int size)
{
   bool res=true;
   for (int i=0;i<size;i++)
   {
      res&=(yh[i]==yd[i]);
      if (i<10) printf("%f %f\n",yh[i],yd[i]);
   }
   return res;
}

/*
   Main Routine: 
   Input: i,[threads]
   Berechnet A*x=y auf der GPU wobei A eine Größe von R^{n x n} hat, mit
   n=1024*i
*/
int main(int argc, char**argv)
{
    int i = 1;
    int t = 512;
    // if (argc>1)
    // {
    //    i=atoi(argv[1]);
    //    if (argc>2) t=atoi(argv[2]);
    // }
    // else 
    // {
    //    printf("Usage: %s i [threads] \n",argv[0]);
    //    return -1;
    // }
    // printf("size %i \n", i);
    int size = 1024 * i;
    //Datenfelder anlegen für Host
    DTYPE *a_host, *yd_host, *yh_host, *x_host;
    //und Device
    DTYPE *a_dev, *y_dev, *x_dev;
    //Events für die Zeitmessung
    cudaEvent_t start, end;
    //Zeiten: 
    //htd: Host->Device Memcpy von A und x
    float htd_time = 0.0;
    //dth: Device->Host Memcpy von y
    float dth_time = 0.0;
    //kernelA, kernelAT
    float kernelA_time = 0.0;
    float kernelAT_time = 0.0;

    //TODO: Host Speicher anlegen und A und x füllen
    a_host = (DTYPE*)malloc(size * size * sizeof(DTYPE));
    x_host = (DTYPE*)malloc(size * sizeof(DTYPE));
    yd_host = (DTYPE*)malloc(size * sizeof(DTYPE));
    yh_host = (DTYPE*)malloc(size * sizeof(DTYPE));

    fillA(a_host, size);
    fillX(x_host, size);

    //TODO: CUDA Events erstellen
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    //TODO: CUDA Speicher anlegen für alle Arrays (a_dev,x_dev,y_dev)
    cudaMalloc((void**)&a_dev, size*size * sizeof(DTYPE));
    cudaMalloc((void**)&x_dev, size * sizeof(DTYPE));
    cudaMalloc((void**)&y_dev, size * sizeof(DTYPE));

    //TODO: Host->Device Memcpy von A und x + Zeitmessung
    cudaEventRecord(start, 0);
    cudaMemcpy(x_dev, x_host, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(a_dev, a_host, size*size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&htd_time, start, end);

    //Konfiguration der CUDA Kernels
    dim3 threads(512);
    dim3 grid(size, size / threads.x);

    //TODO: kernelAx ausführen und Zeit messen
    cudaEventRecord(start, 0);
    kernelAx<<<grid, threads>>>(a_dev, x_dev, y_dev, size);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&kernelA_time, start, end);

    //TODO: Device->Host Memcpy für y_dev -> yd_host
    cudaEventRecord(start, 0);
    cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&dth_time, start, end);

    printf("GPU timing in ms: h->d: %f kernelAx: %f kernelATx: %f d->h: %f\n", htd_time, kernelA_time, kernelAT_time, dth_time);

    hostAx(a_host, x_host, yh_host, size);
    //TODO: checkResult aufrufen
    checkResult(yh_host, yd_host, size);

    //TODO: Speicher freigeben (Host UND Device)
    cudaFree(a_dev);
    cudaFree(x_dev);
    cudaFree(y_dev);

    free(a_host);
    free(x_host);
    free(yh_host);
    free(yd_host);
}
