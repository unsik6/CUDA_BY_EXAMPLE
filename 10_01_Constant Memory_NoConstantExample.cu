#include <iostream>

#ifndef __CUDACC__
#define __CUDACC__
#include <device_functions.h>
#endif

#include "cuda.h"
#include "cuda_runtime.h"
#include "...\CUDA-training-master\utils\cuda_by_example\common\cpu_bitmap.h"
#include "...\CUDA-training-master\utils\cuda_by_example\common\cpu_anim.h"


static void HandleError(cudaError_t, const char*, int);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define INF 2e10f
#define DIM 1024

#define rnd(x) (x *rand()/RAND_MAX)
#define SPHERES 20

struct Sphere
{
	float r, b, g;	// 구의 색
	float radius;	// 구의 반지름
	float x, y, z;	// 구의 중심좌표

	// (ox, oy)의 픽셀에서 발사된 광선이 이 구와 교차되는지 계산
	// 만약 광선이 구와 교차한다면 광선과 구가 닿는 위치까지의 거리를 계산
	__device__ float hit(float ox, float oy, float* n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if (dx * dx + dy * dy < radius * radius)
		{
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};

Sphere* s;

__global__ void kernel(Sphere *s, unsigned char* ptr)
{
	// threadIdx/blockIdx로 픽셀 위치 결정
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);

	// 광선과 구의 교차여부 확인
	float r = 0, g = 0, b = 0;
	float maxz = -INF;
	for (int i = 0; i < SPHERES; i++)
	{
		float n;
		float t = s[i].hit(ox, oy, &n);

		// 이전에 확인한 구보다 더 가까운 구라면 저장한다.
		if (t > maxz)
		{
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
		}
	}

	// for문을 통해 모든 구 중에 가장 가까운 구를 찾았으므로 해당 구의 색상을 이미지에 저장한다.
	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(b * 255);
	ptr[offset * 4 + 2] = (int)(r * 255);
	ptr[offset * 4 + 3] = 255;

}

int main(void)
{
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;

	// 비트맵 GPU메모리 할당
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));;

	// 구 데이터를 위한 GPU메모리 할당
	HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES));

	// 임시 메모리를 할당하고 그것을 초기화한 후에 GPU의 메모리로 복사후 임시 메모리를 해제한다.
	// 각 구의 정보들은 랜덤으로 할당한다.
	Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	for (int i = 0; i < SPHERES; i++)
	{
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;
	}
	HANDLE_ERROR(cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));
	free(temp_s);

	// 비트맵 생성
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <grids, threads >> > (s, dev_bitmap);

	// GPU -> CPU
	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	// 출력
	bitmap.display_and_exit();

	// 메모리 해제
	cudaFree(dev_bitmap);
	cudaFree(s);

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