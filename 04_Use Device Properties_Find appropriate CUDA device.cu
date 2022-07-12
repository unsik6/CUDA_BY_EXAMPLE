#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int main(void)
{
	/*	����̽��� ������ ���� ���Ǹ� �̿��Ͽ� �� ȿ������ ����̽��� ã�ų� �����Ͽ� ���α׷��� ������ �� �ִ�.
	*	ex)
	*		- ���� ����̽� �� ����� ������ ���� ����̽��� ���ϰ� �ʹ�.
	*		- Ŀ���� CPU�� ����� ��ġ���� ��ȣ�ۿ��� �ʿ䰡 ���� ��, CPU�� �ý��� �޸𸮸� �����ϴ� ������ GPU���� �ڵ带 �����ϰ� �ʹ�.
	*		- Ư�� ������ Ư�� ���������� �����ϴ�. �ش��ϴ� ������ ���� ����̽��� ã�ƾ� �Ѵ�.
	*/

	/*
	* ����: �����е� �ε� �Ҽ����� �ʿ��� ���ø����̼� �ۼ��ϰ� �ִٰ� ����
	*		�����е� �ε� �Ҽ��� ������ 1.3���� �̻��� CUDA ��� �ɷ��� ������ �׷��� ī�尡 �����Ѵ�.
	*		�ش��ϴ� ������ ����̽��� �ּ� 1�� ã�ƾ� �Ѵ�.
	* 
	*		����: �� ������ 1���� ũ�ų�, �� ������ 1�̰� �� ������ 3 �̻��� ����̽��� ã�´�.
	*/

	// '03_Query of Device Properties.cu'������ ���� ��� ����̽��� ��ȸ�ϴ� ����� ����� �� �ִ�.
	// �Ʒ��� ����� CUDA ��Ÿ�ӿ� ���� �ڵ�ȭ�� ����̴�.
	
	// __host____device__ cudaError_t cudaGetDevice(int* device)
	// ���� ��� ���� ����̽��� ��ȣ�� *device�� �����Ѵ�.
	// __host__ cudaError_t cudaSetDevice(int dev);
	// GPU ���࿡ ����� ����̽��� device ��ȣ�� ���� ����̽��� �����Ѵ�.

	int dev;

	// ���� ��� ���� ����̽� ��ȣ ��������
	HANDLE_ERROR(cudaGetDevice(&dev));
	printf("ID of current CUDA device: %d\n", dev);

	// cudaDeviceProp ����ü�� ����̽��� �������� �ϴ� �Ӽ����� ä���.
	cudaDeviceProp prop;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 3;
	
	// __host__ cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop);
	/*
	*	- �Ķ����:
	*		1) *device: ���� ������ ����̽��� ��ȣ
	*		2) *prop: ã���� �ϴ� ����
	*/

	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	printf("ID of CUDA device closest to revision 1.3 : %d\n", dev);
	HANDLE_ERROR(cudaSetDevice(dev));

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