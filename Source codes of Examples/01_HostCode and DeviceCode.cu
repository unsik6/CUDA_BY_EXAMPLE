#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// host code
/*
int main(void)
{
	printf("Hello, World!\n");
	return 0;
}
*/

// device code
/*
* __global__ ���ľ�� ���� ��� NVCC�� �ش� �Լ��� ����̽� �ڵ带 �ٷ�� �����Ϸ����� �����Ѵ�.
* ���� ���� �Ű��������� �Լ� ���η� ������ ���޵��� �ʴ´�.
* �ٸ�, runtime �ý��ۿ� �Ѱ����� runtime�� ����̽� �ڵ带 ��� ����(launch)�� �������� ������ ��ģ��.
*/
__global__ void kernel(void) {

}

int main(void)
{
	kernel <<<1, 1 >>> ();
	printf("Hello, World!\n");
	return 0;
}