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
* __global__ 수식어구가 붙은 경우 NVCC가 해당 함수를 디바이스 코드를 다루는 컴파일러에게 전달한다.
* 꺽쇠 안의 매개변수들은 함수 내부로 실제로 전달되지 않는다.
* 다만, runtime 시스템에 넘겨져서 runtime이 디바이스 코드를 어떻게 개시(launch)할 것인지에 영향을 끼친다.
*/
__global__ void kernel(void) {

}

int main(void)
{
	kernel <<<1, 1 >>> ();
	printf("Hello, World!\n");
	return 0;
}