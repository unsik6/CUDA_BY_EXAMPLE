#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int main(void)
{
	/*	cuda ���μ����� ���� ��Ⱑ ���� ���� �� �ִ�.
	*	����, cuda ���μ����� ������ ���;� �� ���� �ִ�.
	*	
	*	__host____device__ cudaError_t cudaGetDeviceCount(int* count)
	*		- �Ű������� ���޹��� *count�� ��� �ɷ��� 2.0 �̻��� ����̽��� ���� ��ȯ�Ѵ�.
	*		- return: cudaSuccess
	* 
	*	cudaGetDeviceCount()�� ȣ���� �� �� ����̽����� �ݺ��ϸ鼭 ����̽���� ���õ� ������ ������ �� �ִ�.
	*	CUDA ��Ÿ���� cudaDeviceProp ����ü�� ���� ���� �Ӽ��� ��ȯ�Ѵ�. (API ������ ��: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)
	*/

	// ��ȸ�� ����̽��� ������ ���� ����ü
	cudaDeviceProp prop;
	
	// cuda ���μ����� ���� ����̽��� ���� ���� ����
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));


	// count�� ����� ���� ���ؼ� ����̽��� ������ ��ȸ�Ѵ�.
	for (int i = 0; i < count; i++)
	{
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));

		//...���⼭ ����...//

		// �Ϲ� ����
		printf("--- General Information for device %d ---\n", i);

		// char Name[256]
			printf("Name: %s\n", prop.name);

		// Cuda ��� �ɷ� ����: (��: CUDA 1.3 ������ major�� 1, minor�� 3)
		// int major: ����̽� ��� �ɷ�(Compute Capability)�� �� ���� ��ȣ
		// int minor: ����̽� ��� �ɷ�(Compute Capability)�� �� ���� ��ȣ
			printf("Compute capability: %d, %d\n", prop.major, prop.minor);

		// int clockRate: Ŭ�� ���ļ� (Khz)
			printf("Clock rate: %d\n", prop.clockRate);

		// int deviceOverplap (boolean ��): �� ����̽��� cudaMemcpy()�� Ŀ�ο��� ���ÿ�(concurrently) ������ �� �ִ��� !!! ����� ������ �ʱ� ������ asyncEngineCount�� ��� ����Ѵ�. !!!
		// int asyncEngineCount: �񵿱� ������ ��
			printf("Device cpy overlap: ");
			if (prop.deviceOverlap) printf("Enabled\n");
			else printf("Disabled\n");

		// int kernelExecTimeoutEnabled (boolean ��): ����̽����� ����Ǵ� Ŀ�ο� ��Ÿ���� ������ �ִ��� ��Ÿ����.
			printf("Kernel execition timeout : ");
			if(prop.kernelExecTimeoutEnabled) printf("Enabled\n");
			else printf("Disabled\n");

		// �޸� ����
		printf("--- Memory Information for device %d ---\n", i);

		// size_t totalGlobalMem: ����̽� ���� �޸��� �� (bytes)
			printf("Total global Mem: %1d\n", prop.totalGlobalMem);

		// size_t totalConstMem: �̿밡���� ��� �޸��� ũ�� (bytes)
			printf("Total constant Mem: %1d\n", prop.totalConstMem);

		// size_t memPitch: �޸� ���纻���� ȣ��Ǵ� �ִ� ��ġ (bytes)
			printf("Max Mem pitch: %1d\n", prop.memPitch);

		// size_t textureAlignment: �ؽ�ó ���Ŀ� ���� ����̽��� �䱸����
			printf("Texture Alignment: %1d\n", prop.textureAlignment);

		// ��Ƽ ���μ��� ����
		printf("--- MP Information for device %d ---\n", i);

		// int multiProcessorCount: ��Ƽ���μ��� ����
			printf("Multiprocessor count: %d\n", prop.multiProcessorCount);

		// size_t sharedMemPerBlock: ���(block) �� �̿� ������ �����޸��� �ִ� �� (bytes)
			printf("Shared mem per mp: %1d\n", prop.sharedMemPerBlock);

		// int regsPerBlock: ���(block) �� �̿� ������ 32-bit ���������� ��
			printf("Registers per mp: %d\n", prop.regsPerBlock);

		// int warpSize: �ϳ��� ����(warp)�� ���� �������� ����
		// warp�� NVDIA�� ����, thread�� ����̴�. (AMD�� ��� wavefront��� ��)
			printf("Threads in warp: %d\n", prop.warpSize);

		// int maxThreadsPerBlock: �ϳ��� ����� ������ �� �ִ� �������� �ִ� ����
			printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

		// int maxThreadsDim[3]: �ϳ��� ��Ͽ��� �� ����(dimension)�� ���� �� �ִ� �������� �ִ� ���� (X, Y, Z)
			printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

		// int maxGridSize[3]: �ϳ��� �׸���(grid)���� �� ������ ���� �� �ִ� ����� �ִ� ����
			printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
			
			printf("\n");
	}

	return 0;
}


// ���� �߻��� ��� �� �����ϴ� �Լ� - å ������ ����.
static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}