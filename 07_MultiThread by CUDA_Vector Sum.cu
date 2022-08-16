#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "E:\00_NEW_ERA\01_INHA\00_TCLAB\07_CUDA\CUDA_PRACTICE_01\CUDA_PRACTICE_01\CUDA-training-master\utils\cuda_by_example\common\cpu_bitmap.h"


static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// Case 1: 1�� Block & N�� Thread
// 1���� Block�� N���� Thread�� �̿��� ����
// ���� ���������� N���� Block�� ���� 1���� Thread�� �̿��ߴ�.
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

	// GPU ����̽� �޸� �Ҵ�
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// CPU���� a�� b ä���
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	// �迭 a�� b�� ����̽� �޸𸮿� ����
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	add << <1, N >> > (dev_a, dev_b, dev_c);

	// �迭 c�� �ٽ� host(CPU) �޸𸮷� ����
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++)
		printf("%d + %d = %d\n", a[i], b[i], c[i]);

	// GPU ����̽� �޸� ����
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

	return 0;
}
*/


// Case 2: Multi blocks & Multi threads - Thread �� ���� ����ϱ�
// Thread�� ���� cudaDeviceProp.maxThreadsPerBlock�� ���� ���ѵȴ�.
// �׷��Ƿ� �� ���� ���� ����� �����ϱ� ���ؼ� �� Block �� ������ Thread�� ���� �����ϰ�, ���� ���� Block�� ����Ѵ�.
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

	// GPU ����̽� �޸� �Ҵ�
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// CPU���� a�� b ä���
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	// �迭 a�� b�� ����̽� �޸𸮿� ����
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	// Block�� ���� ������ ������ ���� ����ȴ�.
	// �� ��, �������� �״�� �����Ͽ� Block�� ������ ���, �ʿ��� ������ �� ���� ���� Thread���� ����ϰ� �ȴ�.
	// ����, �ܼ��� N/128���� Block�鿡 ���Ͽ� 128���� Thread�� �̿��� ��� N�� 130�̶��, ����ϴ� �� Thread�� ���� 128�� �ȴ�.
	// �׷��Ƿ� (N + 127)/128�� ���� ������ �ø��� �����Ѵ�.
	add << <(N+127)/128, 128 >> > (dev_a, dev_b, dev_c);

	// �迭 c�� �ٽ� host(CPU) �޸𸮷� ����
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++)
		printf("%d + %d = %d\n", a[i], b[i], c[i]);

	// GPU ����̽� �޸� ����
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

	return 0;
}
*/

// Case 3: Multi blocks & Multi threads - Block�� Thread �� ���� ����ϱ�

// Block�� ���� Grid�� ������ cudaDeviceProp.maxGridSize�� �ʰ��� �� ����.
// �׷��Ƿ� ��� ���� Block�� ��� ���� Thread���� ������ �ھ�� �Ͽ� �ݺ����� ���� ��� ������ ó���� �� �ִ�.
// �̷� ������ ������ ���, �� Thread���� ����ǰ� �ִ� �����층�� �ϵ����� �����ϵ��� �Ѵ�.
// �ϵ��� �����ϴ� ������κ��� ����ȭ�� �и�(Decoupling the parallelzation)�ϴ� ���� CUDA C�� �۾��� ���Եȴ�.
// �� ��쿡�� ���͵��� ũ�Ⱑ ����̽� �޸��� �� ũ��(cudaDeviceProp.totalConstMem)�� �ʰ������� ������ �ȴ�.

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

	// GPU ����̽� �޸� �Ҵ�
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// CPU���� a�� b ä���
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	// �迭 a�� b�� ����̽� �޸𸮿� ����
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	add << <128, 128 >> > (dev_a, dev_b, dev_c);

	// �迭 c�� �ٽ� host(CPU) �޸𸮷� ����
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++)
		printf("%d + %d = %d\n", a[i], b[i], c[i]);

	// GPU ����̽� �޸� ����
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

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