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

const int threadsPerBlock = 32;

__global__ void kernelReduceGroupShfl(DTYPE *a, DTYPE *y, int size)
{
    size_t index_src = (threadIdx.y + blockDim.y * blockIdx.y) * size + blockIdx.x * blockDim.x + threadIdx.x;
    size_t index_dst = (threadIdx.y + blockDim.y * blockIdx.y) * size + blockIdx.x;

    thread_block_tile<threadsPerBlock> tile = tiled_partition<threadsPerBlock>(this_thread_block());

    DTYPE val = a[index_src];

    for (int k = blockDim.x / 2; k > 0; k >>= 1)
    {
        val += tile.shfl_down(val, k);
    }
    
    if (threadIdx.x == 0)
    {
        if (gridDim.x == 1)
        {
            y[blockIdx.y * blockDim.y + threadIdx.y] = val;
        }
        else
        {
            a[index_dst] = val;
        }
    }
}

__global__ void kernelReduceGroup(DTYPE *a, DTYPE *y, int size)
{

    __shared__ DTYPE cache[threadsPerBlock * threadsPerBlock];

    size_t index_src = (threadIdx.y + blockDim.y * blockIdx.y) * size + blockIdx.x * blockDim.x + threadIdx.x;
    size_t index_dst = (threadIdx.y + blockDim.y * blockIdx.y) * size + blockIdx.x;

    thread_block_tile<threadsPerBlock> tile = tiled_partition<threadsPerBlock>(this_thread_block());

    cache[threadIdx.x + threadIdx.y * threadsPerBlock] = a[index_src];

    for (int k = blockDim.x / 2; k > 0; k >>= 1)
    {
        if (threadIdx.x < k)
        {
            cache[threadIdx.x + threadIdx.y * threadsPerBlock] += cache[threadIdx.x + threadIdx.y * threadsPerBlock + k];
        }
        tile.sync();
    }
    if (threadIdx.x == 0)
    {
        if (gridDim.x == 1)
        {
            y[blockIdx.y * blockDim.y + threadIdx.y] = cache[0];
        }
        else
        {
            a[index_dst] = cache[0];
        }
    }
}

__global__ void kernelReduceSM(DTYPE *a, DTYPE *y, int size)
{

    __shared__ DTYPE cache[threadsPerBlock * threadsPerBlock];

    size_t index_src = (threadIdx.y + blockDim.y * blockIdx.y) * size + blockIdx.x * blockDim.x + threadIdx.x;
    size_t index_dst = (threadIdx.y + blockDim.y * blockIdx.y) * size + blockIdx.x;

    cache[threadIdx.x + threadIdx.y * threadsPerBlock] = a[index_src];

    for (int k = blockDim.x / 2; k > 0; k >>= 1)
    {
        if (threadIdx.x < k)
        {
            cache[threadIdx.x + threadIdx.y * threadsPerBlock] += cache[threadIdx.x + threadIdx.y * threadsPerBlock + k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        if (gridDim.x == 1)
        {
            y[blockIdx.y * blockDim.y + threadIdx.y] = cache[0];
        }
        else
        {
            a[index_dst] = cache[0];
        }
    }
}

__device__ DTYPE multAx(DTYPE *a, DTYPE *x, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    return a[row * size + col] * x[col];
}

__global__ void kernelSMAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    __shared__ DTYPE cache[threadsPerBlock * threadsPerBlock];

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
    }
}

__global__ void kernelSimpleAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    atomicAdd(&y[row], a[row * size + col] * x[col]);
}

__device__ DTYPE multATx(DTYPE *a, DTYPE *x, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    return a[col * size + row] * x[col];
}

__global__ void kernelSMATx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    __shared__ DTYPE cache[threadsPerBlock * threadsPerBlock];

    cache[threadIdx.y * blockDim.y + threadIdx.x] = multATx(a, x, size);
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
    }
}

__global__ void kernelSimpleATx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    atomicAdd(&y[row], a[col * size + row] * x[col]);
}


//Fill A (here with ones)
void fillA(DTYPE *a, int size)
{
    for (int i = 0;i<size*size;i++)
        a[i] = 1.0;
}

//Fill X
void fillX(DTYPE *x, int size)
{
    for (int i = 0;i<size;i++)
        x[i] = (DTYPE)(i + 1);
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
    bool res = true;
    for (int i = 0;i<size;i++)
    {
        res &= (yh[i] == yd[i]);
        if (i<10) printf("%f %f\n", yh[i], yd[i]);
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
    int i = 2;
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
    // 48kB shared / 16kB local
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

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
    dim3 threads(threadsPerBlock, threadsPerBlock);
    dim3 grid(size / threads.x, size / threads.y);

#if 0
    // execute kernelAx and measure Performance
    cudaEventRecord(start, 0);
    
    kernelSMAx << <grid, threads >> >(a_dev, x_dev, y_dev, size);

    int values_to_reduce = size / threadsPerBlock;
    int threads_x = min(values_to_reduce, threadsPerBlock);
    threads = dim3(threads_x, threadsPerBlock);
    int grid_x = values_to_reduce / threads.x;

    while (values_to_reduce > 1)
    {
        grid = dim3(grid_x, size / threads.y);

        kernelReduceGroupShfl << <grid, threads >> >(a_dev, y_dev, size);

        values_to_reduce = grid_x;
        threads_x = min(values_to_reduce, threadsPerBlock);
        threads = dim3(threads_x, threadsPerBlock);
        grid_x = values_to_reduce / threads.x;
    }

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

#else
    // execute kernelAx and measure Performance
    cudaEventRecord(start, 0);
    kernelSMAx << <grid, threads >> >(a_dev, x_dev, y_dev, size);
    cudaMemcpy(copy_a_dev_to_host, a_dev, size*size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

    int values_to_reduce = size / threadsPerBlock;
    int threads_x = min(values_to_reduce, threadsPerBlock);
    threads = dim3(threads_x, threadsPerBlock);
    int grid_x = values_to_reduce / threads.x;

    while (values_to_reduce > 1)
    {
        grid = dim3(grid_x, size / threads.y);
        kernelReduceGroupShfl << <grid, threads >> >(a_dev, y_dev, size);
        values_to_reduce = grid_x;
        threads_x = min(values_to_reduce, threadsPerBlock);
        threads = dim3(threads_x, threadsPerBlock);
        grid_x = values_to_reduce / threads.x;

    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&kernelAT_time, start, end);

    // Device->Host Memcpy for y_dev -> yd_host
    cudaEventRecord(start, 0);
    cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    // Check Ax result
    hostATx(a_host, x_host, yh_host, size);
    checkResult(yh_host, yd_host, size);
    printf("\n");

    cudaEventElapsedTime(&dth_time, start, end);


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
