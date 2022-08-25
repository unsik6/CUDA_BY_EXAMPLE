#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int main(void)
{
	/*	디바이스의 정보에 대한 질의를 이용하여 더 효율적인 디바이스를 찾거나 설정하여 프로그램을 실행할 수 있다.
	*	ex)
	*		- 여러 디바이스 중 우수한 성능을 갖는 디바이스를 택하고 싶다.
	*		- 커널이 CPU와 가까운 위치에서 상호작용할 필요가 있을 때, CPU와 시스템 메모리를 공유하는 통합형 GPU에서 코드를 실행하고 싶다.
	*		- 특정 연산은 특정 버전에서만 가능하다. 해당하는 버전을 갖는 디바이스를 찾아야 한다.
	*/

	/*
	* 예제: 배정밀도 부동 소수점이 필요한 어플리케이션 작성하고 있다고 가정
	*		배정밀도 부동 소수점 연산은 1.3버전 이상의 CUDA 계산 능력을 보유한 그래픽 카드가 지원한다.
	*		해당하는 버전의 디바이스를 최소 1개 찾아야 한다.
	* 
	*		문제: 주 버전이 1보다 크거나, 주 버전이 1이고 부 버전이 3 이상인 디바이스를 찾는다.
	*/

	// '03_Query of Device Properties.cu'에서와 같이 모든 디바이스를 순회하는 방법을 사용할 수 있다.
	// 아래의 방법은 CUDA 런타임에 의한 자동화된 방법이다.
	
	// __host____device__ cudaError_t cudaGetDevice(int* device)
	// 현재 사용 중인 디바이스의 번호를 *device에 저장한다.
	// __host__ cudaError_t cudaSetDevice(int dev);
	// GPU 실행에 사용할 디바이스를 device 번호를 갖는 디바이스로 설정한다.

	int dev;

	// 현재 사용 중인 디바이스 번호 가져오기
	HANDLE_ERROR(cudaGetDevice(&dev));
	printf("ID of current CUDA device: %d\n", dev);

	// cudaDeviceProp 구조체에 디바이스가 가졌으면 하는 속성들을 채운다.
	cudaDeviceProp prop;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 3;
	
	// __host__ cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop);
	/*
	*	- 파라미터:
	*		1) *device: 가장 적합한 디바이스의 번호
	*		2) *prop: 찾고자 하는 조건
	*/

	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	printf("ID of CUDA device closest to revision 1.3 : %d\n", dev);
	HANDLE_ERROR(cudaSetDevice(dev));

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