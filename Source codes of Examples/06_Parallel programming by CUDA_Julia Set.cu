#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "...\CUDA-training-master\utils\cuda_by_example\common\cpu_bitmap.h"


static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define DIM 1000

// 복소수 구조체
struct cuComplex
{
	float r;
	float i;
	__device__ cuComplex(float a, float b) : r(a), i(b) {}
	__device__ float magnitude2(void) { return r * r + i * i; }
	
	__device__ cuComplex operator*(const cuComplex& a)
	{
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r + a.r, i + a.i);
	}
};


// Julia Set인지 판단하는 함수
// 해당 지점이 Julia Set이라면 1을 반환하고 아니면 0을 반환한다.
// 기본적으로 CPUBitmap이라는 2차원 픽셀 공간을 복소수 공간으로 치환하여 계산한다.
__device__ int julia(int x, int y)
{
	// 확대 및 축소를 위한 스케일
	const float scale = 1.5;

	// 이미지 공간의 중심에서 복소수 공간의 중심으로 이동하기 위해 DIM / 2 만큼 이동한 후, 이미지의 범위가 -1.0~1.0을 보장하기 위해 DIM / 2를 나눈 것이다.
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	/////	Julia Function /////
	/// Z(n+!) = Z(n)^2 + C	///
	///////////////////////////

	// 임의의 복소수 C: 이 C의 값에 따라 줄리아 세트 모양이 달라진다.
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++)
	{
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}
	return 1;
}

__global__ void kernel(unsigned char* ptr)
{
	// threadIdx/blockIDx로 픽셀 위치를 결정한다.

	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;	//gridDim은 그리드의 차원 수를 저장하는 상수이다.

	// Julia Set인지 판단하는 함수를 통해 점을 찍는다.
	int juliaValue = julia(x, y);
	ptr[offset * 4 + 0] = 255 * juliaValue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

int main(void)
{
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;

	// 비트맵 사이즈의 디바이스 메모리 할당
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

	// DIM * DIM 크기의 2차원 그리드
	// dim3는 3차원 요소들의 집합이다. CUDA runtime이 dim3를 예상하고 있기 때문에 2차원임에도 이 데이터타입을 사용한다. (현재는 3차원 그리드가 지원되지 않는다.)
	// 2개의 파라미터만 전달되므로 마지막 파라미터는 1이다.
	dim3 grid(DIM, DIM);
	kernel <<< grid, 1 >>> (dev_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	bitmap.display_and_exit();

	cudaFree(dev_bitmap);

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
