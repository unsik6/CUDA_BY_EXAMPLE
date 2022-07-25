#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "E:\00_NEW_ERA\01_INHA\00_TCLAB\07_CUDA\CUDA_PRACTICE_01\CUDA_PRACTICE_01\CUDA-training-master\utils\cuda_by_example\common\cpu_bitmap.h"


static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define DIM 1000

// ���Ҽ� ����ü
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


// Julia Set���� �Ǵ��ϴ� �Լ�
// �ش� ������ Julia Set�̶�� 1�� ��ȯ�ϰ� �ƴϸ� 0�� ��ȯ�Ѵ�.
// �⺻������ CPUBitmap�̶�� 2���� �ȼ� ������ ���Ҽ� �������� ġȯ�Ͽ� ����Ѵ�.
__device__ int julia(int x, int y)
{
	// Ȯ�� �� ��Ҹ� ���� ������
	const float scale = 1.5;

	// �̹��� ������ �߽ɿ��� ���Ҽ� ������ �߽����� �̵��ϱ� ���� DIM / 2 ��ŭ �̵��� ��, �̹����� ������ -1.0~1.0�� �����ϱ� ���� DIM / 2�� ���� ���̴�.
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	/////	Julia Function /////
	/// Z(n+!) = Z(n)^2 + C	///
	///////////////////////////

	// ������ ���Ҽ� C: �� C�� ���� ���� �ٸ��� ��Ʈ ����� �޶�����.
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
	// threadIdx/blockIDx�� �ȼ� ��ġ�� �����Ѵ�.

	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;	//gridDim�� �׸����� ���� ���� �����ϴ� ����̴�.

	// Julia Set���� �Ǵ��ϴ� �Լ��� ���� ���� ��´�.
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

	// ��Ʈ�� �������� ����̽� �޸� �Ҵ�
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

	// DIM * DIM ũ���� 2���� �׸���
	// dim3�� 3���� ��ҵ��� �����̴�. CUDA runtime�� dim3�� �����ϰ� �ֱ� ������ 2�����ӿ��� �� ������Ÿ���� ����Ѵ�. (����� 3���� �׸��尡 �������� �ʴ´�.)
	// 2���� �Ķ���͸� ���޵ǹǷ� ������ �Ķ���ʹ� 1�̴�.
	dim3 grid(DIM, DIM);
	kernel <<< grid, 1 >>> (dev_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	bitmap.display_and_exit();

	cudaFree(dev_bitmap);

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