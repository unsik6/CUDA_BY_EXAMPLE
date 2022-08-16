#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "E:\00_NEW_ERA\01_INHA\00_TCLAB\07_CUDA\CUDA_PRACTICE_01\CUDA_PRACTICE_01\CUDA-training-master\utils\cuda_by_example\common\cpu_bitmap.h"


static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// Case 1: 1개 Block & N개 Thread
// 1개의 Block에 N개의 Thread를 이용한 예제
// 이전 예제에서는 N개의 Block에 각각 1개의 Thread를 이용했다.
/*
#define N 10

__global__ void add(int* a, int* b, int* c)
{
	int tid = threadIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main(void)
{
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	// GPU 디바이스 메모리 할당
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// CPU에서 a와 b 채우기
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	// 배열 a와 b를 디바이스 메모리에 복사
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	add << <1, N >> > (dev_a, dev_b, dev_c);

	// 배열 c를 다시 host(CPU) 메모리로 복사
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++)
		printf("%d + %d = %d\n", a[i], b[i], c[i]);

	// GPU 디바이스 메모리 해제
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

	return 0;
}
*/


// Case 2: Multi blocks & Multi threads - Thread 수 제한 고려하기
// Thread의 수가 cudaDeviceProp.maxThreadsPerBlock에 의해 제한된다.
// 그러므로 더 많은 수의 명령을 수행하기 위해서 한 Block 당 수행할 Thread의 수를 제한하고, 여러 개의 Block을 사용한다.
/*
#define N 10

__global__ void add(int* a, int* b, int* c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main(void)
{
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	// GPU 디바이스 메모리 할당
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// CPU에서 a와 b 채우기
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	// 배열 a와 b를 디바이스 메모리에 복사
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	// Block의 수는 나눗셈 연산을 통해 진행된다.
	// 이 때, 나눗셈을 그대로 진행하여 Block을 배정할 경우, 필요한 수보다 더 적은 수의 Thread들을 사용하게 된다.
	// 만약, 단순히 N/128개의 Block들에 대하여 128개의 Thread를 이용할 경우 N이 130이라면, 사용하는 총 Thread의 수는 128이 된다.
	// 그러므로 (N + 127)/128을 통해 나눗셈 올림을 진행한다.
	add << <(N+127)/128, 128 >> > (dev_a, dev_b, dev_c);

	// 배열 c를 다시 host(CPU) 메모리로 복사
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++)
		printf("%d + %d = %d\n", a[i], b[i], c[i]);

	// GPU 디바이스 메모리 해제
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

	return 0;
}
*/

// Case 3: Multi blocks & Multi threads - Block과 Thread 수 제한 고려하기

// Block에 대한 Grid의 차원은 cudaDeviceProp.maxGridSize를 초과할 수 없다.
// 그러므로 상수 개의 Block과 상수 개의 Thread들을 각각의 코어로 하여 반복문을 통해 산술 연산을 처리할 수 있다.
// 이런 식으로 구현할 경우, 각 Thread에서 실행되고 있는 스케쥴링은 하드웨어에서 진행하도록 한다.
// 하드웨어가 실행하는 방법으로부터 병렬화를 분리(Decoupling the parallelzation)하는 일은 CUDA C의 작업에 포함된다.
// 이 경우에는 벡터들의 크기가 디바이스 메모리의 총 크기(cudaDeviceProp.totalConstMem)를 초과하지만 않으면 된다.

#define N (33 * 1024)

__global__ void add(int* a, int* b, int* c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	while (tid < N)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

int main(void)
{
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	// GPU 디바이스 메모리 할당
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// CPU에서 a와 b 채우기
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	// 배열 a와 b를 디바이스 메모리에 복사
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	add << <128, 128 >> > (dev_a, dev_b, dev_c);

	// 배열 c를 다시 host(CPU) 메모리로 복사
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++)
		printf("%d + %d = %d\n", a[i], b[i], c[i]);

	// GPU 디바이스 메모리 해제
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

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