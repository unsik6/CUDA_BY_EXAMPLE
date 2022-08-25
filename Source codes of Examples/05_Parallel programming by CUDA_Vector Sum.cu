#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define N 10

__global__ void add(int* a, int* b, int* c)
{
	int tid = blockIdx.x;	// Compute the data of this index.
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}


int main(void)
{
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	// allocate GPU memory
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// fill 'a' and 'b' in CPU
	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	// copy 'a' and 'b' to GPU memory
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	add <<<N, 1 >>> (dev_a, dev_b, dev_c);

	// Copy the result of 'add', 'c' from GPU memory to Host(CPU) memory
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	// output result
	for (int i = 0; i < N; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	// deallocate GPU memory
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