#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int main(void)
{
	/*	cuda 프로세서를 갖는 기기가 여러 개일 수 있다.
	*	또한, cuda 프로세서의 정보를 얻어와야 할 수도 있다.
	*	
	*	__host____device__ cudaError_t cudaGetDeviceCount(int* count)
	*		- 매개변수로 전달받은 *count에 계산 능력이 2.0 이상인 디바이스의 수를 반환한다.
	*		- return: cudaSuccess
	* 
	*	cudaGetDeviceCount()를 호출한 후 각 디바이스들을 반복하면서 디바이스들과 관련된 정보를 질의할 수 있다.
	*	CUDA 런타임은 cudaDeviceProp 구조체를 통해 여러 속성을 반환한다. (API 참조할 것: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)
	*/

	// 조회할 디바이스의 정보를 담을 구조체
	cudaDeviceProp prop;
	
	// cuda 프로세서를 갖는 디바이스의 수를 담을 변수
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));


	// count에 저장된 값을 통해서 디바이스의 정보를 순회한다.
	for (int i = 0; i < count; i++)
	{
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));

		//...여기서 질의...//

		// 일반 정보
		printf("--- General Information for device %d ---\n", i);

		// char Name[256]
			printf("Name: %s\n", prop.name);

		// Cuda 계산 능력 버전: (예: CUDA 1.3 버전은 major가 1, minor가 3)
		// int major: 디바이스 계산 능력(Compute Capability)의 주 개정 번호
		// int minor: 디바이스 계산 능력(Compute Capability)의 부 개정 번호
			printf("Compute capability: %d, %d\n", prop.major, prop.minor);

		// int clockRate: 클럭 주파수 (Khz)
			printf("Clock rate: %d\n", prop.clockRate);

		// int deviceOverplap (boolean 값): 이 디바이스가 cudaMemcpy()와 커널에서 동시에(concurrently) 수행할 수 있는지 !!! 현재는 사용되지 않기 때문에 asyncEngineCount를 대신 사용한다. !!!
		// int asyncEngineCount: 비동기 엔진의 수
			printf("Device cpy overlap: ");
			if (prop.deviceOverlap) printf("Enabled\n");
			else printf("Disabled\n");

		// int kernelExecTimeoutEnabled (boolean 값): 디바이스에서 실행되는 커널에 런타임의 제한이 있는지 나타낸다.
			printf("Kernel execition timeout : ");
			if(prop.kernelExecTimeoutEnabled) printf("Enabled\n");
			else printf("Disabled\n");

		// 메모리 정보
		printf("--- Memory Information for device %d ---\n", i);

		// size_t totalGlobalMem: 디바이스 전역 메모리의 양 (bytes)
			printf("Total global Mem: %1d\n", prop.totalGlobalMem);

		// size_t totalConstMem: 이용가능한 상수 메모리의 크기 (bytes)
			printf("Total constant Mem: %1d\n", prop.totalConstMem);

		// size_t memPitch: 메모리 복사본에서 호용되는 최대 피치 (bytes)
			printf("Max Mem pitch: %1d\n", prop.memPitch);

		// size_t textureAlignment: 텍스처 정렬에 대한 디바이스의 요구사항
			printf("Texture Alignment: %1d\n", prop.textureAlignment);

		// 멀티 프로세서 정보
		printf("--- MP Information for device %d ---\n", i);

		// int multiProcessorCount: 멀티프로세서 개수
			printf("Multiprocessor count: %d\n", prop.multiProcessorCount);

		// size_t sharedMemPerBlock: 블록(block) 당 이용 가능한 공유메모리의 최대 양 (bytes)
			printf("Shared mem per mp: %1d\n", prop.sharedMemPerBlock);

		// int regsPerBlock: 블록(block) 당 이용 가능한 32-bit 레지스터의 수
			printf("Registers per mp: %d\n", prop.regsPerBlock);

		// int warpSize: 하나의 워프(warp)가 갖는 스레드의 개수
		// warp란 NVDIA식 용어로, thread의 덩어리이다. (AMD의 경우 wavefront라고 함)
			printf("Threads in warp: %d\n", prop.warpSize);

		// int maxThreadsPerBlock: 하나의 블록이 포함할 수 있는 스레드의 최대 개수
			printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

		// int maxThreadsDim[3]: 하나의 블록에서 각 차원(dimension)이 가질 수 있는 스레드의 최대 개수 (X, Y, Z)
			printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

		// int maxGridSize[3]: 하나의 그리드(grid)에서 각 차원이 가질 수 있는 블록의 최대 개수
			printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
			
			printf("\n");
	}

	return 0;
}


// 에러 발생시 출력 후 종료하는 함수 - 책 예제에 포함.
static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}