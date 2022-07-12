#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 에러 발생시 출력 후 종료하는 함수 - 책 예제에 포함.
static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void add(int a, int b, int* c) {
	*c = a + b;
}

int main(void)
{
	int c;
	int* dev_c;

	// __host___device__ cudaError_t cudaMalloc(void** 새로 할당할 메모리의 주소를 가리키는 포인터, size_t 할당할 메모리의 크기)
	/* -CUDA 런타임이 디바이스 메모리를 할당한다.해당 메모리를 가리키는 포인터는 첫번째 인자에 저장된다.
	* - return: cudaError_t 열거체
	* 	 0: cudaSuccess: 성공.
	* 	 1: cudaErrorInvalidValue: 파라미터의 값이 범위를 초과하여 실패.
	* 	 2: cudaErrorMemoryAllocation: 할당할 메모리 공간이 너무 작아서 실패.
	*  
	*  - 호스트에서 실행되는 코드에서 cudaMalloc()에 의해 반환되는 포인터(디바이스 포인터)를 역참조해서는 안된다.
	* 	 포인터의 위치 이동, 포인터를 이용한 연산, 포인터의 타입 변환은 가능
	* 	 포인터의 메모리를 읽거나 기록하기 위해 쓸 수 없다.
	*  - cudaMalloc()으로 할당한 메모리 포인터를 디바이스에서 실행되는 함수로 전달할 수 있다.
	* 	 디바이스에서 실행되는 코드에서 cudaMalloc()으로 할당한 메모리 포인터를 이용하여 메모리를 읽거나 쓸 수 있다.
	* 	 cudaMalloc()으로 할당한 메모리 포인터를 호스트에서 실행되는 함수로 전달할 수 있다. 다만, 호스트에서 실행되는 코드에서 cudaMalloc()으로 할당한 메모리 포인터를 이용하여 읽거나 쓸 수 없다.
	*/
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));


	// 디바이스 메모리에 접근하는 보편적인 방식 1
	add <<<1, 1 >> > (2, 7, dev_c);

	// 디바이스 메모리에 접근하는 보편적인 방식 2
	// __host__ cudaError_t cudaMemcpy(src, dst, cudaMemcpyKind)
	/* 세번째 인자는 src와 dst가 각각 어디에 속하는지 나타낸다.
	* - return: cudaMalloc과 동일
	* - cudaMemcpyKind 열거체
	*	0: cudaMemcpyHostToHost: 호스트에서 호스트	// 단, 이 경우에는 그냥 C의 memcpy를 사용하면 된다.
	*	1: cudaMemcpyHostToDevice: 호스트에서 디바이스
	*	2: cudaMemcpyDeviceToHost: 디바이스에서 호스트
	*	3: cudaMecmcpyDeviceToDevice: 디바이스에서 디바이스
	*	4: cudaMemcpyDefault: src 포인터가 무엇이냐에 따라 추론된다.
	* 
	*/
	HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

	printf("2 + 7 = %d\n", c);

	// cudaMalloc()으로 할당한 메모리는 C의 free()로 해제할 수 없고
	// cudaFree()를 통해 해제한다.
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