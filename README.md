
# CUDA_BY_EXAMPLE

- The text book, 'CUDA BY EXAMPLE' 
- The major reference: [NVIDA CUDA Toolkit Documentation v11.7.0](https://docs.nvidia.com/cuda/)

- The environment
	> CUDA version: CUDA 11.7
	> OS: Window 10 64-bits
	> GPU: NVIDIA GeForce GTX 1060 6GB

Unfortunately, All comments or descriptions of source codes are written by korean to improve the efficiency of studying.

## Contents
1. [00_start from Jun 27, 2022](#00_start-from-June-27,-2022)
2. [01_Host Code and Device Code](#01_Host-Code-and-Device-Code)
3. [02_Allocating and Using Device Memory](#02_Allocating-and-Using-Device-Memory)
4. 

## 00_start from June 27, 2022
Install CUDA and construct the development environment.
- Issue: ERROR E0029
&nbsp;&nbsp;This is a common error, Syntax error by the third '<' that is not correct to standard C++. Many solution is suggested in STACK OVERFLOW and NVIDIA Forum. However the dominating solution is 'no solution'. <br/>
referece: [visual studio .cu file shows syntax error but compiles successfully](https://stackoverflow.com/questions/15205458/visual-studio-cu-file-shows-syntax-error-but-compiles-successfully)

## [01_Host Code and Device Code](https://github.com/unsik6/CUDA_BY_EXAMPLE/blob/main/01_HostCode%20and%20DeviceCode.cu)
- keywords: host code, device code, \_\_global\_\_
### 1. What is host code and device code?
- <b>Host code</b> is the code execute on CPU; 
- <b>Device code</b> is the code execute on GPU for parallel prongramming;
 <br/>
 
### 2. What is <i>\_\_global\_\_</i> keword?
```C
__global__ void kernel(void){}
int main(void) {
	kernel<<<1, 1>>>();
	return 0;
}
```

- The functions that declared <i>\_\_global\_\_</i> is delivered to the compiler that deal with 'device code', such as NVCC(NVDIA CUDA Compiler)
- The parameters in "<<< >>>" are not passed into the function. The parameters are passed to CUDA Runtime, and affect how to launch the device codes by CUDA Runtime.
- We can make the parameters in "()" passed into the function, like we have done in C/C++.

## [02_Allocating and Using Device Memory](https://github.com/unsik6/CUDA_BY_EXAMPLE/blob/main/02_Allocating%20and%20Using%20Device%20Memory.cu)
- keywords: malloc, cudaMalloc, free, cudaFree, memcpy, cudaMemcpy

### 1. \_\_host\_\_\_\_device\_\_ cudaError_t cudaMalloc((void** devPtr, size_t size)
```c
int* dev_c;
cudaMalloc((void**)&dev_c, sizeof(int));
```
- <i>cudaMalloc()</i>: CUDA Runtime allocates the device memory, whose size is <i>size</i>(bytes), to <i>dev_c</i>. 
- Host codes never dereferece the pointer that points the device memory to read or write. Moving the position of pointer, operating using the pointer and transformating the type of pointer are possible. But, device codes can do.
-  The pointer allocated by <i>cudaMalloc()</i> can be passed into the function that executes in whether device or host.


### 2. \_\_host\_\_ cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
```c
int c;
int* dev_c;
cudaMalloc((void**)&dev_c, sizeof(int));
cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
```
- <i>cudaMemcpy()</i>: copies <i>count</i> bytes from the memory area pointed to by <i>src</i> to the memory area pointed to by <i>dst</i>.
- <i>cudaMemcpyKind</i> is the enum that means direction of copy.
	> 0: cudaMemcpyHostToHost	// It is equal to call <i>memcpy()</i> of C in host.
	> 1: cudaMemcpyHostToDevice
	> 2: cudaMemcpyDeviceToHost
	> 3: cudaMecmcpyDeviceToDevice
	> 4: cudaMemcpyDefault: Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing
- Host codes can use the variable of device code, <i>dev_c</i> by using <i>cudaMemcpy()</i>.

### 3. \_\_host\_\_\_\_device\_\_ cudaError_t cudaFree(void* devPtr)
```C
cudaFree(dev_c);
```
- The device memory allocated by <i>cudaMalloc()</i> can be deallocated by <i>cudaFree()</i>, not <i>free()</i> of C.
