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

const int tileSize = 32;

__device__ DTYPE multAx(DTYPE *a, DTYPE *x, int size)
{
    int row = blockIdx.x;
    int col = threadIdx.x + blockIdx.y * blockDim.x;

    return a[row * size + col] * x[col];
}

__device__ DTYPE multATx(DTYPE *a, DTYPE *x, int size)
{
    int col = blockIdx.x;
    int row = blockDim.x * blockIdx.y + threadIdx.x;

    return a[row * size + col] * x[row];
}

__device__ DTYPE reduce(thread_group g, DTYPE *cache, DTYPE val)
{
    int gtid_x = g.thread_rank();
    cache[gtid_x] = val;
    g.sync();

    //printf("gtidx %d cache %f \n", gtid_x, cache[gtid_x]);
    
    // #pragma unroll
    for (int k = g.size() / 2; k > 0; k >>= 1)
    {
       
        if (gtid_x < k)
        {
            cache[gtid_x] += cache[gtid_x + k];
        }
        g.sync();
    }

    //printf("cache[0] %f \n", cache[0]);
    return cache[0];
}

__device__ int reduce_sum_shfl(thread_block_tile<tileSize> g, int val)
{
    // Each thread adds sum[i] to sum[delta+i]
    for (int i = g.size() / 2; i > 0; i /= 2) 
    {
        val += g.shfl_down(val, i);
    }

    return val;
}

__global__ void kernelSimpleAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    int row = blockIdx.x;
    int col = blockDim.x * blockIdx.y + threadIdx.x;

    atomicAdd(&y[row], a[row * size + col] * x[col]);
}

__global__ void kernelAxShfl(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    __shared__ DTYPE cache[maxThreadsPerBlock];

    DTYPE val = multAx(a, x, size);

    thread_group g = this_thread_block();
    int tileIdx = g.thread_rank() / 32;
    DTYPE* t = &cache[32 * tileIdx];

    thread_block_tile<tileSize> tile = tiled_partition<tileSize>(this_thread_block());

    DTYPE sum = reduce_sum_shfl(tile, val);

    if (tile.thread_rank() == 0) atomicAdd(&y[blockIdx.x], sum);
}

__global__ void kernelAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    __shared__ DTYPE cache[maxThreadsPerBlock];

    DTYPE val = multAx(a, x, size);

    thread_group g = this_thread_block();
    int tileIdx = g.thread_rank() / 32;
    DTYPE* t = &cache[32 * tileIdx];

    thread_group tile = tiled_partition(g, 32);

    DTYPE sum = reduce(tile, t, val);

    if (tile.thread_rank() == 0) atomicAdd(&y[blockIdx.x], sum);
}


__global__ void kernelSimpleATx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    int col = blockIdx.x;
    int row = blockDim.x * blockIdx.y + threadIdx.x;

    atomicAdd(&y[row], a[row * size + col] * x[row]);
}


__global__ void kernelATx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    __shared__ DTYPE cache[maxThreadsPerBlock];

    DTYPE val = multATx(a, x, size);

    thread_group g = this_thread_block();
    int tileIdx = g.thread_rank() / 32;
    DTYPE* t = &cache[32 * tileIdx];

    thread_group tile = tiled_partition(g, 32);

    DTYPE sum = reduce(tile, t, val);

    if (tile.thread_rank() == 0) atomicAdd(&y[blockIdx.x], sum);
}

__global__ void kernelATxShfl(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    __shared__ DTYPE cache[maxThreadsPerBlock];

    DTYPE val = multATx(a, x, size);

    thread_group g = this_thread_block();
    int tileIdx = g.thread_rank() / 32;
    DTYPE* t = &cache[32 * tileIdx];

    thread_block_tile<tileSize> tile = tiled_partition<tileSize>(this_thread_block());

    DTYPE sum = reduce_sum_shfl(tile, val);

    if (tile.thread_rank() == 0) atomicAdd(&y[blockIdx.x], sum);
}



//Fill A (here with ones)
void fillA(DTYPE *a, int size)
{
   for (int i=0;i<size*size;i++)
      a[i]=1.0;
}

//Fill X
void fillX(DTYPE *x, int size)
{
   for (int i=0;i<size;i++)
      x[i]= (DTYPE)(i+1);
}

void hostAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    // Compute A*x=y on host
    for (unsigned int i = 0; i < size; i++)
    {
        y[i] = 0;
        for (unsigned int j = 0; j < size; j++)
        {
            y[i] += a[j * size + i] * x[j];
        }
    }
}

void hostATx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    // Compute AT*x=y on host
    for (unsigned int i = 0; i < size; i++)
    {
        y[i] = 0;
        for (unsigned int j = 0; j < size; j++)
        {
            y[i] += a[i * size + j] * x[j];
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
   Compute A*x=y on the GPU where A is of size R^{n x n} with
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
    // Create data arrays for host
    DTYPE *a_host, *yd_host, *yh_host, *x_host;
    //and device
    DTYPE *a_dev, *y_dev, *x_dev;
    //Events for performance measurement
    cudaEvent_t start, end;
    //Zeiten: 
    //htd: Host->Device Memcpy from A and x
    float htd_time = 0.0;
    //dth: Device->Host Memcpy from y
    float dth_time = 0.0;
    //kernelA, kernelAT
    float kernelA_time = 0.0;
    float kernelAT_time = 0.0;

    // Allocate Host Memory and fill A and x
    a_host = (DTYPE*)malloc(size * size * sizeof(DTYPE));
    x_host = (DTYPE*)malloc(size * sizeof(DTYPE));
    yd_host = (DTYPE*)malloc(size * sizeof(DTYPE));
    yh_host = (DTYPE*)malloc(size * sizeof(DTYPE));

    fillA(a_host, size);
    fillX(x_host, size);

    // Set CUDA cache config
    // 16kB shared / 48kB local
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    // 48kB shared / 16kB local
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    // Create CUDA Events
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Allocate CUDA memory for all arrays (a_dev,x_dev,y_dev)
    cudaMalloc((void**)&a_dev, size*size * sizeof(DTYPE));
    cudaMalloc((void**)&x_dev, size * sizeof(DTYPE));
    cudaMalloc((void**)&y_dev, size * sizeof(DTYPE));

    // Host->Device Memcpy from A and x + performance measurement
    cudaEventRecord(start, 0);
    cudaMemcpy(x_dev, x_host, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(a_dev, a_host, size*size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&htd_time, start, end);

    // Configurate CUDA kernel
    dim3 threads(t);
    dim3 grid(size, size / threads.x);
#if 1
    // execute kernelAx and measure Performance
    cudaEventRecord(start, 0);
    //kernelAx<<<grid, threads>>>(a_dev, x_dev, y_dev, size);
    kernelAxShfl<<<grid, threads>>>(a_dev, x_dev, y_dev, size);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&kernelA_time, start, end);

    // Device->Host Memcpy for y_dev -> yd_host
    cudaEventRecord(start, 0);
    cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    // Check Ax result
    hostAx(a_host, x_host, yh_host, size);
    checkResult(yh_host, yd_host, size);
    printf("\n");

    cudaEventElapsedTime(&dth_time, start, end);

    
    ////////////////////////////////////////////////////////////////////////
#else
    // execute kernelATx and measure Performance
    cudaEventRecord(start, 0);
    //kernelATx <<<grid, threads >>>(a_dev, x_dev, y_dev, size);
    kernelATxShfl <<<grid, threads >>>(a_dev, x_dev, y_dev, size);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&kernelAT_time, start, end);

    // Device->Host Memcpy for y_dev -> yd_host
    cudaEventRecord(start, 0);
    cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&dth_time, start, end);

    // Check ATx result
    hostATx(a_host, x_host, yh_host, size);
    checkResult(yh_host, yd_host, size);
    printf("\n");
    #endif

    printf("GPU timing in ms: h->d: %f kernelAx: %f kernelATx: %f d->h: %f\n", htd_time, kernelA_time, kernelAT_time, dth_time);


    // Free memory (Host and Device)
    cudaFree(a_dev);
    cudaFree(x_dev);
    cudaFree(y_dev);

    free(a_host);
    free(x_host);
    free(yh_host);
    free(yd_host);
}
