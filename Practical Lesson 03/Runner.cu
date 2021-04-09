#include <cudaDefs.h>
#include <iostream>

using namespace std;
constexpr unsigned int THREADS_PER_BLOCK_DIM = 8;				//=64 threads in block

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();
cudaEvent_t start, stop;

__global__ void fillData(const unsigned int pitch, const unsigned int rows, const unsigned int cols, float *data)
{	
		int current_col = blockIdx.x * blockDim.x + threadIdx.x;
		int current_row = blockIdx.y * blockDim.y + threadIdx.y;
		
		if ((current_col >= cols) || (current_row >= rows)) 
			return;

		data[current_row * (pitch / sizeof(float)) + current_col] = current_row * cols + current_col;
}

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	//Adresa na GPU 
	float* devPtr;
	//Velkost naalokovanej pamete
	size_t pitch;

	const unsigned int mRows = 5; //3
	const unsigned int mCols = 10; //3

	//TODO: Allocate Pitch memory
	cudaMallocPitch(&devPtr, &pitch, mCols * sizeof(float), mRows); // pitch 512
	//std::cout << pitch;

	//TODO: Prepare grid, blocks
	dim3 blockSize = dim3(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM, 1); // 8x8
	dim3 gridSize = dim3((mCols + THREADS_PER_BLOCK_DIM - 1) / THREADS_PER_BLOCK_DIM, (mRows + THREADS_PER_BLOCK_DIM - 1) / THREADS_PER_BLOCK_DIM, 1);

	//TODO: Call kernel
	fillData <<<gridSize, blockSize >> > (pitch, mRows, mCols, devPtr);

	//TODO: Allocate Host memory and copy back Device data
	float* hostMemory = (float*)malloc(mRows * mCols * sizeof(float));
	cudaMemcpy2D(hostMemory, mCols * sizeof(float), devPtr, pitch, mCols * sizeof(float), mRows, cudaMemcpyDeviceToHost);

	//TODO: Check data
		for (int i = 0; i < mRows * mCols; i++)
		{
			if (i % (mCols) == 0)
			{
				cout << endl;
			}
			cout << hostMemory[i] << " ";
		}
	

	//TODO: Free memory
	free(hostMemory);
	cudaFree(devPtr);

	return 0;
}
