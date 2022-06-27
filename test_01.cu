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
__global__ void kernel(void) {

}

int main(void)
{
	kernel <<<1, 1 >>> ();
	printf("Hello, World!\n");
	return 0;
}