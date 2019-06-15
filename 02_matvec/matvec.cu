#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <cooperative_groups.h>
using namespace cooperative_groups;

// CUDA runtime
#include <cuda_runtime.h>

#define DTYPE float

const int maxThreadsPerBlock = 32;

const int tileSize = 32;

// __device__ DTYPE multAx(DTYPE *a, DTYPE *x, int size)
// {
//     int row = blockIdx.x;
//     int col = threadIdx.x + blockIdx.y * blockDim.x;

//     return a[row * size + col] * x[col];
// }

__device__ DTYPE multAx(DTYPE *a, DTYPE *x, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    return a[row * size + col] * x[col];
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

// __global__ void kernelGroupShflAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
// {
//     __shared__ DTYPE cache[maxThreadsPerBlock];

//     DTYPE val = multAx(a, x, size);

//     thread_group g = this_thread_block();
//     int tileIdx = g.thread_rank() / 32;
//     DTYPE* t = &cache[32 * tileIdx];

//     thread_block_tile<tileSize> tile = tiled_partition<tileSize>(this_thread_block());

//     DTYPE sum = reduce_sum_shfl(tile, val);

//     if (tile.thread_rank() == 0) atomicAdd(&y[blockIdx.x], sum);
// }

// __global__ void kernelGroupAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
// {
//     __shared__ DTYPE cache[maxThreadsPerBlock];

//     DTYPE val = multAx(a, x, size);

//     thread_group g = this_thread_block();
//     int tileIdx = g.thread_rank() / 32;
//     DTYPE* t = &cache[32 * tileIdx];

//     thread_group tile = tiled_partition(g, 32);

//     DTYPE sum = reduce(tile, t, val);

//     if (tile.thread_rank() == 0) atomicAdd(&y[blockIdx.x], sum);
// }


__global__ void kernelReduceSimple(DTYPE *a, DTYPE *y, int size)
{

    __shared__ DTYPE cache[maxThreadsPerBlock * maxThreadsPerBlock];
    size_t index_src = (threadIdx.y + blockDim.y * blockIdx.y) * size + blockIdx.x * blockDim.x + threadIdx.x;
    size_t index_dst = (threadIdx.y + blockDim.y * blockIdx.y) * size + blockIdx.x;


    cache[threadIdx.x + threadIdx.y * maxThreadsPerBlock] = a[index_src];

    // printf("a %f \n", cache[threadIdx.x + threadIdx.y * maxThreadsPerBlock]);


    for (int k = blockDim.x / 2; k > 0; k >>= 1)
    {
        if (threadIdx.x < k)
        {
            cache[threadIdx.x + threadIdx.y * maxThreadsPerBlock] += cache[threadIdx.x  + threadIdx.y * maxThreadsPerBlock + k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) 
    {
        // printf("a %f \n", cache[0]);

        if (gridDim.x == 1)
        {
            // printf("a %f \n", a[(threadIdx.y + blockDim.y * blockIdx.y) * size + threadIdx.x]);
            // printf("a %f \n", cache[0]);

            y[blockIdx.y * blockDim.y + threadIdx.y] = cache[0];
        }
        else{
            a[index_dst] = cache[0];
        }
    }
}



__global__ void kernelSMAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    __shared__ DTYPE cache[maxThreadsPerBlock * maxThreadsPerBlock];

    cache[threadIdx.y * blockDim.y + threadIdx.x] = multAx(a, x, size);
    __syncthreads();

    for (int k = blockDim.x / 2; k > 0; k >>= 1)
    {
        if (threadIdx.x < k)
        {
            cache[threadIdx.y * blockDim.y + threadIdx.x] += cache[threadIdx.y * blockDim.y + threadIdx.x + k];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) 
    {
        a[(blockIdx.y * blockDim.y + threadIdx.y) * size + blockIdx.x] = cache[0];
        // atomicAdd(&y[blockIdx.y * blockDim.y + threadIdx.y], cache[0]);
    }
}


// __global__ void kernelSimpleAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
// {
//     int row = blockIdx.x;
//     int col = blockDim.x * blockIdx.y + threadIdx.x;

//     atomicAdd(&y[row], a[row * size + col] * x[col]);
// }


__global__ void kernelSimpleAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    atomicAdd(&y[row], a[row * size + col] * x[col]);
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
    int t = 4;
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
    // int size = 64;
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
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    // Create CUDA Events
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Allocate CUDA memory for all arrays (a_dev,x_dev,y_dev)
    cudaMalloc((void**)&a_dev, size*size * sizeof(DTYPE));
    cudaMalloc((void**)&x_dev, size * sizeof(DTYPE));
    cudaMalloc((void**)&y_dev, size * sizeof(DTYPE));
    
    DTYPE* copy_a_dev_to_host = (DTYPE*)malloc(size * size * sizeof(DTYPE));



    // Host->Device Memcpy from A and x + performance measurement
    cudaEventRecord(start, 0);
    cudaMemcpy(x_dev, x_host, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(a_dev, a_host, size*size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&htd_time, start, end);

    // Configurate CUDA kernel
    dim3 threads(t,t);
    dim3 grid(size / threads.x, size / threads.y);

    kernelSMAx<<<grid, threads>>>(a_dev, x_dev, y_dev, size);

    cudaMemcpy(copy_a_dev_to_host, a_dev, size*size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

    // for(unsigned y = 0; y != size; ++y){
    //     std::cout <<  std::endl << y << ": ";
    std::cout <<  std::endl;
         for(unsigned x = 0; x != size; ++x){

             size_t index = x;
             std::cout << copy_a_dev_to_host[index] << " ";
         }
    // }


    int values_to_reduce = size / t;
    int threads_x = min(values_to_reduce, t);
    threads = dim3(threads_x, t);
    // threads = dim3(8, t);
    int grid_x = values_to_reduce / threads.x;

    printf("\nvalues to reduce %d grid %d \n", values_to_reduce, grid_x);

    while (values_to_reduce > 1)
    {
        grid = dim3(grid_x, size / threads.y);

        printf("values to reduce %d grid_x %d s threads.x %d\n", values_to_reduce, grid.x, threads.x);

        kernelReduceSimple<<<grid, threads>>>(a_dev, y_dev, size);
        cudaMemcpy(copy_a_dev_to_host, a_dev, size*size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

        values_to_reduce = grid_x;
        threads_x = min(values_to_reduce, t);
        threads = dim3(threads_x, t);
        grid_x = values_to_reduce / threads.x;
        
        std::cout <<  std::endl;
        for(unsigned x = 0; x != size; ++x){
             size_t index = x;
             std::cout << copy_a_dev_to_host[index] << " ";
        }
        std::cout <<  std::endl;

    }

// #if 1
//     // execute kernelAx and measure Performance
//     cudaEventRecord(start, 0);
//     //kernelSimpleAx<<<grid, threads>>>(a_dev, x_dev, y_dev, size);
//     kernelSMAx<<<grid, threads>>>(a_dev, x_dev, y_dev, size);
//     //kernelGroupAx<<<grid, threads>>>(a_dev, x_dev, y_dev, size);
//     //kernelGroupShflAx<<<grid, threads>>>(a_dev, x_dev, y_dev, size);
//     cudaEventRecord(end, 0);
//     cudaEventSynchronize(end);

//     cudaEventElapsedTime(&kernelA_time, start, end);

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
