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
	// ���� �޸��� ���, �ϳ��� Block �ȿ� �����ϴ� ��� Thread���� �����ϴ� �޸��̴�.
	// �Ʒ��� �ڵ忡���� �����޸��� cache �迭(Block �� Thread ���� ��ŭ�� ����)�� �� ��ҵ��� �� Thread�� ����Ѵ�.
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	// �� Thread���� ���� ������ ���� ������ �յ��� ��� ���Ѵ�.
	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	// �� Thread���� �����޸� �迭�� ��� �� �ڽ��� ����ϴ� ��ҿ� ���� ���� �κ� ���� �����Ѵ�.
	cache[cacheIndex] = temp;

	// �� Block�� Thread���� ����ȭ��Ų��.
	// #ifndef __CUDACC__
	// #define __CUDACC__
	//		#include <device_functions.h>
	// #endif
	// �� ��ó�� ������ �־���� �Ѵ�.
	__syncthreads();

	// ������
	// ���� �ڵ� ������, �������� ���ؼ��� threadPerBlock�� 2�� ������� �Ѵ�.
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];

		// �������� �� ������ ������ ������ �Ǿ����� ����ȭ�� �ʿ��ϴ�.
		// * ���� ����ȭ�� ���ؼ� �� ����ȭ �Լ��� ���� if �� �ȿ� �ִ´ٸ�, �ϳ��� ������ Ȱ���Ͽ� SW������ ����ȭ�� �����ϴ� ������� ���ؼ� Progress ���谡 ��Ÿ���� �Ͱ� ���� ������ �߻��ϰ� �ȴ�.
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

	// CPU���� ����� �޸𸮸� �Ҵ��Ѵ�.
	a = (float*)malloc(N * sizeof(float));
	b = (float*)malloc(N * sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid * sizeof(float));	// �κ� ���� ��� ��� ������ ���Ǿ� ����Ǹ� ������ Grid�� ����ϱ� ������ Grid �� Block�� �� ��ŭ �ʿ��ϴ�.

	// GPU ����̽� �޸𸮸� �Ҵ��Ѵ�.
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float))); // �κ� ���� ��� ��� ������ ���Ǿ� ����Ǹ� ������ Grid�� ����ϱ� ������ Grid �� Block�� �� ��ŭ �ʿ��ϴ�.

	// ȣ��Ʈ �޸𸮿� �����͸� ä���.
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = 2 * i;
	}

	// ����̽� �޸𸮿� ȣ��Ʈ �޸𸮸� �����Ѵ�.
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

	// ���� ����
	dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);

	// �κ��� �迭�� ����̽����� ȣ��Ʈ�� �����Ѵ�.
	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float),cudaMemcpyDeviceToHost));

	// CPU���� �κ��� �迭�� ���� ���� ������ ����� ����Ѵ�.
	// �� ������ �������� ���� ��������, CPU���� �����ϴ� ������ GPU�� ����ϱ⿡�� ������ ���� �ſ� ���� �����̴�.
	c = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		c += partial_c[i];
	}

#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
	printf("Does GPU value %.6g = %6.g?\n", c, 2 * sum_squares((float)(N - 1)));

	// �޸� ����
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
	free(a);
	free(b);
	free(partial_c);

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