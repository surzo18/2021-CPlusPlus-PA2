#include <cudaDefs.h>
#include <iostream>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();
cudaEvent_t start, stop;

constexpr unsigned int THREADS_PER_BLOCK = 512;
constexpr unsigned int MEMBLOCKS_PER_THREADBLOCK = 2;

using namespace std;

__global__ void add1(const int* __restrict__ a, const int* __restrict__ b, const unsigned int length, int* __restrict__ c)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int skip = gridDim.x * blockDim.x;
    while (offset < length)
    {
        c[offset] = a[offset] + b[offset];
        offset += skip;
    }
}

__global__ void add2(const int* __restrict__ a, const int* __restrict__ b, const unsigned int length, int* __restrict__ c)
{
	//TODO: c[i] = a[i] + b[i]
	for (int i = 0; i << length; i++)
	{
		c[i] = a[i] + b[i];
	}
}

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	constexpr unsigned int length = 1<<20;
	constexpr unsigned int sizeInBytes = length * sizeof(int);	

	//TODO: Allocate Host memory
	int* h_a = new int[length];
	int* h_b = new int[length];
	int* h_c = new int[length];


	//TODO: Init data
	for (int i = 0; i < length; i++)
	{
		h_a[i] = i+1;
		h_b[i] = i+1;
	}


	//TODO: Allocate Device memory
	int* d_a, * d_b, *d_c;
	cudaMalloc((void**)&d_a, sizeInBytes);
	cudaMalloc((void**)&d_b, sizeInBytes);
	cudaMalloc((void**)&d_c, sizeInBytes);


	//TODO: Copy Data
	cudaMemcpy(d_a, h_a, sizeInBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeInBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, sizeInBytes, cudaMemcpyHostToDevice);

	//TODO: Prepare grid and blocks
	dim3 gridSize(length / THREADS_PER_BLOCK, 1, 1); // ASK WHY
	dim3 blockSize(THREADS_PER_BLOCK, 1, 1); //128,256,512,1024

	//TODO: Call kernel
	cudaEventRecord(start); //Start event counter
	add1<<<gridSize, blockSize>>> (d_a, d_b, length, d_c);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaMemcpy(h_c, d_c, sizeInBytes, cudaMemcpyDeviceToHost);
	
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	cout << "TIME: " << ms << endl;

	//TODO: Check results
	for (int i = 0; i < 10; i++)
	{
		cout << h_c[i] << " ";
	}
	cout << endl;

	//TODO: Free memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	delete[] h_a;
	delete[] h_b;
	delete[] h_c;
}
