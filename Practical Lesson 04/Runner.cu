#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <random>
#include<iostream>

using namespace std;

//WARNING!!! Do not change TPB and NO_FORCES for this demo !!!
constexpr unsigned int TPB = 128;
constexpr unsigned int NO_FORCES = 256;
constexpr unsigned int NO_RAIN_DROPS = 1 << 20;

constexpr unsigned int MEM_BLOCKS_PER_THREAD_BLOCK = 8;

constexpr int FLOAT_MIN = 0;
constexpr int FLOAT_MAX = 100;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace std;

__host__ float3* createData(const unsigned int length)
{
	//TODO: Generate float3 vectors. You can use 'make_float3' method.
	float3* data = new float3[length];

	for (int i = 0; i < length; i++) {
		//Random generator of float
		std::random_device rd;
		std::default_random_engine eng(rd());
		std::uniform_real_distribution<> distr(FLOAT_MIN, FLOAT_MAX);

		float f1 = distr(eng);
		float f2 = distr(eng);
		float f3 = distr(eng);
		data[i] = make_float3(f1, f2, f3);
	}
	return data;
}

__host__ void printData(const float3* data, const unsigned int length)
{
	if (data == 0) return;
	const float3* ptr = data;
	for (unsigned int i = 0; i < length; i++, ptr++)
	{
		printf("Print Data: %5.2f %5.2f %5.2f \n", ptr->x, ptr->y, ptr->z);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Sums the forces to get the final one using parallel reduction. 
/// 		    WARNING!!! The method was written to meet input requirements of our example, i.e. 128 threads and 256 forces  </summary>
/// <param name="dForces">	  	The forces. </param>
/// <param name="noForces">   	The number of forces. </param>
/// <param name="dFinalForce">	[in,out] If non-null, the final force. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void reduce(const float3* __restrict__ dForces, const unsigned int noForces, float3* __restrict__ dFinalForce)
{
	__shared__ float3 sForces[TPB];					//SEE THE WARNING MESSAGE !!!
	unsigned int x = threadIdx.x;
	unsigned int next = TPB;						//SEE THE WARNING MESSAGE !!!

	volatile float f1 = dForces[x].x + dForces[x + next].x;
	volatile float f2 = dForces[x].y + dForces[x + next].y;
	volatile float f3 = dForces[x].z + dForces[x + next].z;
	sForces[x] = make_float3(f1, f2, f3);
	next /= 2;
	__syncthreads();

	while (next != 1) {
		if (x < next)
		{
			f1 = sForces[x].x + sForces[x + next].x;
			f2 = sForces[x].y + sForces[x + next].y;
			f3 = sForces[x].z + sForces[x + next].z;
			sForces[x] = make_float3(f1, f2, f3);
			next /= 2;
			if ((next / 32) != 1)
				__syncthreads();
		}
		else
		{
			return;
		}

	}
	//normalize
	sForces[0].x /= 100;
	sForces[0].y /= 100;
	sForces[0].z /= 100;
	*dFinalForce = sForces[0];

	//TODO: Make the reduction
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds the FinalForce to every Rain drops position. </summary>
/// <param name="dFinalForce">	The final force. </param>
/// <param name="noRainDrops">	The number of rain drops. </param>
/// <param name="dRainDrops"> 	[in,out] If non-null, the rain drops positions. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void add(const float3* __restrict__ dFinalForce, const unsigned int noRainDrops, float3* __restrict__ dRainDrops)
{
	//TODO: Add the FinalForce to every Rain drops position.
	float3 flt = *dFinalForce;
	unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int skip = gridDim.x * blockDim.x;

	while (offset < noRainDrops)
	{
		dRainDrops[offset].x += flt.x;
		dRainDrops[offset].y += flt.y;
		dRainDrops[offset].z += flt.z;
		offset += skip;
	}

}


int main(int argc, char* argv[])
{

	initializeCUDA(deviceProp);

	/*----------*/
	float test = 4;
	float* d_test;
	float* h_test = new float();
	h_test[0] = 4;
	/*
		test -> d_test -> h_test -> vypis
	*/

	cudaMalloc((void**)&d_test, sizeof(float));
	cudaError(cudaMemcpy(d_test, &test, sizeof(float), cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(h_test, d_test, sizeof(float), cudaMemcpyDeviceToHost));
	
	cout << *h_test << " Ahoj kluci" << endl;
	/*---------*/

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;

	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);


	float3* hForces = createData(NO_FORCES);
	float3* hDrops = createData(NO_RAIN_DROPS);

	float3* dForces = nullptr;
	float3* dDrops = nullptr;
	float3* dFinalForce = nullptr;

	checkCudaErrors(cudaMalloc((void**)&dForces, NO_FORCES * sizeof(float3)));
	checkCudaErrors(cudaMemcpy(dForces, hForces, NO_FORCES * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&dDrops, NO_RAIN_DROPS * sizeof(float3)));
	checkCudaErrors(cudaMemcpy(dDrops, hDrops, NO_RAIN_DROPS * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&dFinalForce, sizeof(float3)));

	KernelSetting ksReduce;

	float3* hFinalForce = new float3();

	//TODO: ... Set ksReduce
	ksReduce.dimBlock = dim3(TPB, 1, 1);
	ksReduce.dimGrid = dim3(1, 1, 1);  

	KernelSetting ksAdd;
	//TODO: ... Set ksAdd
	ksAdd.dimBlock = dim3(512, 1, 1);
	int tmp = std::min(((int)NO_RAIN_DROPS + 511) / 512,1472); //  92 SM po 16 blokov max = 1472
																
	ksAdd.dimGrid = dim3(tmp, 1, 1);

	cudaEventRecord(startEvent, 0);
	for (unsigned int i = 0; i < 1000; i++)
	{
		reduce << <ksReduce.dimGrid, ksReduce.dimBlock >> > (dForces, NO_FORCES, dFinalForce);
		add << <ksAdd.dimGrid, ksAdd.dimBlock >> > (dFinalForce, NO_RAIN_DROPS, dDrops);
	}
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	checkDeviceMatrix<float>((float*)dFinalForce, sizeof(float3), 1, 3, "%5.2f ", "Final force");
	// checkDeviceMatrix<float>((float*)dDrops, sizeof(float3), NO_RAIN_DROPS, 3, "%5.2f ", "Final Rain Drops");

	if (hForces)
		free(hForces);
	if (hDrops)
		free(hDrops);
	free(hFinalForce);

	checkCudaErrors(cudaFree(dForces));
	checkCudaErrors(cudaFree(dDrops));



	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	printf("Time to get device properties: %f ms", elapsedTime);
}
