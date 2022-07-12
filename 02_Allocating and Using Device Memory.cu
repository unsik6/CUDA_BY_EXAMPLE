#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ���� �߻��� ��� �� �����ϴ� �Լ� - å ������ ����.
static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void add(int a, int b, int* c) {
	*c = a + b;
}

int main(void)
{
	int c;
	int* dev_c;

	// __host___device__ cudaError_t cudaMalloc(void** ���� �Ҵ��� �޸��� �ּҸ� ����Ű�� ������, size_t �Ҵ��� �޸��� ũ��)
	/* -CUDA ��Ÿ���� ����̽� �޸𸮸� �Ҵ��Ѵ�.�ش� �޸𸮸� ����Ű�� �����ʹ� ù��° ���ڿ� ����ȴ�.
	* - return: cudaError_t ����ü
	* 	 0: cudaSuccess: ����.
	* 	 1: cudaErrorInvalidValue: �Ķ������ ���� ������ �ʰ��Ͽ� ����.
	* 	 2: cudaErrorMemoryAllocation: �Ҵ��� �޸� ������ �ʹ� �۾Ƽ� ����.
	*  
	*  - ȣ��Ʈ���� ����Ǵ� �ڵ忡�� cudaMalloc()�� ���� ��ȯ�Ǵ� ������(����̽� ������)�� �������ؼ��� �ȵȴ�.
	* 	 �������� ��ġ �̵�, �����͸� �̿��� ����, �������� Ÿ�� ��ȯ�� ����
	* 	 �������� �޸𸮸� �аų� ����ϱ� ���� �� �� ����.
	*  - cudaMalloc()���� �Ҵ��� �޸� �����͸� ����̽����� ����Ǵ� �Լ��� ������ �� �ִ�.
	* 	 ����̽����� ����Ǵ� �ڵ忡�� cudaMalloc()���� �Ҵ��� �޸� �����͸� �̿��Ͽ� �޸𸮸� �аų� �� �� �ִ�.
	* 	 cudaMalloc()���� �Ҵ��� �޸� �����͸� ȣ��Ʈ���� ����Ǵ� �Լ��� ������ �� �ִ�. �ٸ�, ȣ��Ʈ���� ����Ǵ� �ڵ忡�� cudaMalloc()���� �Ҵ��� �޸� �����͸� �̿��Ͽ� �аų� �� �� ����.
	*/
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));


	// ����̽� �޸𸮿� �����ϴ� �������� ��� 1
	add <<<1, 1 >> > (2, 7, dev_c);

	// ����̽� �޸𸮿� �����ϴ� �������� ��� 2
	// __host__ cudaError_t cudaMemcpy(src, dst, cudaMemcpyKind)
	/* ����° ���ڴ� src�� dst�� ���� ��� ���ϴ��� ��Ÿ����.
	* - return: cudaMalloc�� ����
	* - cudaMemcpyKind ����ü
	*	0: cudaMemcpyHostToHost: ȣ��Ʈ���� ȣ��Ʈ	// ��, �� ��쿡�� �׳� C�� memcpy�� ����ϸ� �ȴ�.
	*	1: cudaMemcpyHostToDevice: ȣ��Ʈ���� ����̽�
	*	2: cudaMemcpyDeviceToHost: ����̽����� ȣ��Ʈ
	*	3: cudaMecmcpyDeviceToDevice: ����̽����� ����̽�
	*	4: cudaMemcpyDefault: src �����Ͱ� �����̳Ŀ� ���� �߷еȴ�.
	* 
	*/
	HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

	printf("2 + 7 = %d\n", c);

	// cudaMalloc()���� �Ҵ��� �޸𸮴� C�� free()�� ������ �� ����
	// cudaFree()�� ���� �����Ѵ�.
	cudaFree(dev_c);

	return 0;
}

static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}