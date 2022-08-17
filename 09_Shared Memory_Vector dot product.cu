#include <iostream>

#ifndef __CUDACC__
#define __CUDACC__
	#include <device_functions.h>
#endif

#include "cuda.h"
#include "cuda_runtime.h"
#include "E:\00_NEW_ERA\01_INHA\00_TCLAB\07_CUDA\CUDA_PRACTICE_01\CUDA_PRACTICE_01\CUDA-training-master\utils\cuda_by_example\common\cpu_bitmap.h"
#include "E:\00_NEW_ERA\01_INHA\00_TCLAB\07_CUDA\CUDA_PRACTICE_01\CUDA_PRACTICE_01\CUDA-training-master\utils\cuda_by_example\common\cpu_anim.h"


static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float* a, float* b, float* c)
{
	// 공유 메모리의 경우, 하나의 Block 안에 존재하는 모든 Thread들이 공유하는 메모리이다.
	// 아래의 코드에서는 공유메모리인 cache 배열(Block 당 Thread 개수 만큼의 길이)의 각 요소들을 각 Thread가 사용한다.
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	// 각 Thread에서 각각 배정된 벡터 내적의 합들을 모두 더한다.
	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	// 각 Thread에서 공유메모리 배열의 요소 중 자신이 사용하는 요소에 벡터 내적 부분 합을 저장한다.
	cache[cacheIndex] = temp;

	// 이 Block의 Thread들을 동기화시킨다.
	// #ifndef __CUDACC__
	// #define __CUDACC__
	//		#include <device_functions.h>
	// #endif
	// 를 전처리 과정에 넣어줘야 한다.
	__syncthreads();

	// 리덕션
	// 다음 코드 때문에, 리덕션을 위해서는 threadPerBlock은 2의 멱수여야 한다.
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];

		// 리덕션의 각 과정이 완전히 마무리 되었는지 동기화가 필요하다.
		// * 만약 최적화를 위해서 이 동기화 함수를 위의 if 문 안에 넣는다면, 하나의 변수를 활용하여 SW적으로 동기화를 진행하는 방식으로 인해서 Progress 위배가 나타나는 것과 같은 현상이 발생하게 된다.
		__syncthreads();

		i /= 2;
	}

	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

int main(void)
{
	float* a, * b, c, * partial_c;
	float* dev_a, * dev_b, * dev_partial_c;

	// CPU에서 사용할 메모리를 할당한다.
	a = (float*)malloc(N * sizeof(float));
	b = (float*)malloc(N * sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid * sizeof(float));	// 부분 합의 경우 블록 단위로 계산되어 저장되며 일차원 Grid를 사용하기 때문에 Grid 당 Block의 수 만큼 필요하다.

	// GPU 디바이스 메모리를 할당한다.
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float))); // 부분 합의 경우 블록 단위로 계산되어 저장되며 일차원 Grid를 사용하기 때문에 Grid 당 Block의 수 만큼 필요하다.

	// 호스트 메모리에 데이터를 채운다.
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = 2 * i;
	}

	// 디바이스 메모리에 호스트 메모리를 복사한다.
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

	// 내적 실행
	dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);

	// 부분합 배열을 디바이스에서 호스트로 복사한다.
	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float),cudaMemcpyDeviceToHost));

	// CPU에서 부분합 배열을 통해 벡터 내정의 결과를 계산한다.
	// 이 과정은 리덕션의 최종 과정으로, CPU에서 진행하는 이유는 GPU를 사용하기에는 연산의 수가 매우 적기 때문이다.
	c = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		c += partial_c[i];
	}

#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
	printf("Does GPU value %.6g = %6.g?\n", c, 2 * sum_squares((float)(N - 1)));

	// 메모리 해제
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
	free(a);
	free(b);
	free(partial_c);

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