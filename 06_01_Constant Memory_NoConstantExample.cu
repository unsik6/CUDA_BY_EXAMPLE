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
	float r, b, g;	// ���� ��
	float radius;	// ���� ������
	float x, y, z;	// ���� �߽���ǥ

	// (ox, oy)�� �ȼ����� �߻�� ������ �� ���� �����Ǵ��� ���
	// ���� ������ ���� �����Ѵٸ� ������ ���� ��� ��ġ������ �Ÿ��� ���
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
	// threadIdx/blockIdx�� �ȼ� ��ġ ����
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);

	// ������ ���� �������� Ȯ��
	float r = 0, g = 0, b = 0;
	float maxz = -INF;
	for (int i = 0; i < SPHERES; i++)
	{
		float n;
		float t = s[i].hit(ox, oy, &n);

		// ������ Ȯ���� ������ �� ����� ����� �����Ѵ�.
		if (t > maxz)
		{
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
		}
	}

	// for���� ���� ��� �� �߿� ���� ����� ���� ã�����Ƿ� �ش� ���� ������ �̹����� �����Ѵ�.
	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(b * 255);
	ptr[offset * 4 + 2] = (int)(r * 255);
	ptr[offset * 4 + 3] = 255;

}

int main(void)
{
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;

	// ��Ʈ�� GPU�޸� �Ҵ�
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));;

	// �� �����͸� ���� GPU�޸� �Ҵ�
	HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES));

	// �ӽ� �޸𸮸� �Ҵ��ϰ� �װ��� �ʱ�ȭ�� �Ŀ� GPU�� �޸𸮷� ������ �ӽ� �޸𸮸� �����Ѵ�.
	// �� ���� �������� �������� �Ҵ��Ѵ�.
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

	// ��Ʈ�� ����
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <grids, threads >> > (s, dev_bitmap);

	// GPU -> CPU
	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	// ���
	bitmap.display_and_exit();

	// �޸� ����
	cudaFree(dev_bitmap);
	cudaFree(s);

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