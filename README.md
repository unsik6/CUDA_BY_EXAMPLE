


# CUDA_BY_EXAMPLE

- The text book, Jason Sanders, Edward Kandrot, 'CUDA by Example: An Introduction to General-Purpose GPU Programming<sup>1st</sup>', 2011 
- The major reference: [NVIDA CUDA Toolkit Documentation v11.7.0](https://docs.nvidia.com/cuda/)

- The environment
	> CUDA version: CUDA 11.7 <br/>
	> OS: Window 10 64-bits <br/>
	> GPU: NVIDIA GeForce GTX 1060 6GB <br/>

Unfortunately, All comments or descriptions of source codes are written by korean to improve the efficiency of studying.

# Contents
1. [00_start from Jun 27, 2022](#00_start-from-June-27-2022)
2. [01_Host Code and Device Code](#01_Host-Code-and-Device-Code)
3. [02_Allocating and Using Device Memory](#02_Allocating-and-Using-Device-Memory)
4. [03_Query of Device Properties](#03_Query-of-Device-Properties)
5. [04_Use Device Properties_Find appropriate CUDA device](#04_Use-Device-Properties_Find-appropriate-CUDA-device)
6. [05_Parallel programming by CUDA: Vector Sum](#05_Parallel-programming-by-CUDA_Vector-Sum)

<br/><br/>

# 00_start from June 27 2022
Install CUDA and construct the development environment.
- Issue: ERROR E0029
&nbsp;&nbsp;This is a common error, Syntax error by the third '<' that is not correct to standard C++. Many solution is suggested in STACK OVERFLOW and NVIDIA Forum. However the dominating solution is 'no solution'. <br/>
referece: [visual studio .cu file shows syntax error but compiles successfully](https://stackoverflow.com/questions/15205458/visual-studio-cu-file-shows-syntax-error-but-compiles-successfully)

# [01_Host Code and Device Code](https://github.com/unsik6/CUDA_BY_EXAMPLE/blob/main/01_HostCode%20and%20DeviceCode.cu)
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
- The parameters in "<<< >>>" are not passed into the function. The parameters are passed to CUDA Runtime, and affect how to launch the device codes by CUDA Runtime. (The more detail is in <a href = "#05PP_VS">'05_Parallel programming by CUDA: Vector Sum'</a>)
- We can make the parameters in "()" passed into the function, like we have done in C/C++.

<br/><br/>

# [02_Allocating and Using Device Memory](https://github.com/unsik6/CUDA_BY_EXAMPLE/blob/main/02_Allocating%20and%20Using%20Device%20Memory.cu)
- keywords: cudaError, malloc, cudaMalloc, free, cudaFree, memcpy, cudaMemcpy

### 1. enum cudaError
- The flag that means CUDA error types.
	> 0: cudaSuccess: The API call returned with no errors. <br/>
	> 1: cudaErrorInvalidValue: This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.<br/>
	> 2: cudaErrorMemoryAllocation: The API call failed because it was unable to allocate enough memory to perform the requested operation.<br/>
	> <br/>
	> etc...
	
<br/>

### 2. \_\_host\_\_\_\_device\_\_ cudaError_t cudaMalloc((void** devPtr, size_t size)
```c
int* dev_c;
cudaMalloc((void**)&dev_c, sizeof(int));
```
- <i>cudaMalloc()</i>: CUDA Runtime allocates the device memory, whose size is <i>size</i>(bytes), to <i>dev_c</i>. 
- Host codes never dereferece the pointer that points the device memory to read or write. Moving the position of pointer, operating using the pointer and transformating the type of pointer are possible. But, device codes can do.
-  The pointer allocated by <i>cudaMalloc()</i> can be passed into the function that executes in whether device or host.

<br/>

### 3. \_\_host\_\_ cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
```c
int c;
int* dev_c;
cudaMalloc((void**)&dev_c, sizeof(int));
cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
```
- <i>cudaMemcpy()</i>: copies <i>count</i> bytes from the memory area pointed to by <i>src</i> to the memory area pointed to by <i>dst</i>.
- <i>cudaMemcpyKind</i> is the enum that means direction of copy.
	> 0: cudaMemcpyHostToHost	// It is equal to call <i>memcpy()</i> of C in host.<br/>
	> 1: cudaMemcpyHostToDevice<br/>
	> 2: cudaMemcpyDeviceToHost<br/>
	> 3: cudaMecmcpyDeviceToDevice<br/>
	> 4: cudaMemcpyDefault: Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing
- Host codes can use the variable of device code, <i>dev_c</i> by using <i>cudaMemcpy()</i>.

<br/>

### 4. \_\_host\_\_\_\_device\_\_ cudaError_t cudaFree(void* devPtr)
```C
cudaFree(dev_c);
```
- The device memory allocated by <i>cudaMalloc()</i> can be deallocated by <i>cudaFree()</i>, not <i>free()</i> of C.

<br/><br/>
# [03_Query of Device Properties](https://github.com/unsik6/CUDA_BY_EXAMPLE/blob/main/03_Query%20of%20Device%20Properties.cu)
- keywords: cudaDeviceProp, cudaGetDeviceCount, cudaGetDeviceProperties

### 1. cudaDeviceProp
- <i>cudaDeviceProp</i> is the struct that includes CUDA device properties such as id, name and etc.
- API: [cudaDeviceProp Struct Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)

<br/>

### 2. \_\_host\_\_\_\_device\_\_ cudaError_t cudaGetDeviceCount(int* count)
```C
cudaDeviceProp prop;
int count;
cudaGetDeviceCount(&count)
```
- <i>cudaGetDeviceCount()</i> returns the number of devices with compute capability greater or equal to 2.0 into <i>*count</i>.

<br/>

### 3. \_\_host\_\_ cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device)
```C
cudaDeviceProp prop;
int count;
cudaGetDeviceCount(&count)
for (int i = 0; i < count; i++)
{
	cudaGetDeviceProperties(&prop, i);
	// query //
}
```
- <i>cudaGetDeviceProperties()</i> returns the pointer of cudaDeviceProp of device whose number is <i>device</i> into <i>*prop</i>.
- We can access sequencially all information of each device in loop using the count by <i>cudaGetDeviceCount()</i>.

<br/><br/>
# [04_Use Device Properties_Find appropriate CUDA device](https://github.com/unsik6/CUDA_BY_EXAMPLE/blob/main/04_Use%20Device%20Properties_Find%20appropriate%20CUDA%20device.cu)

- keywords: cudaGetDevice, cudaSetDevice, cudaChooseDevice

### 1. \_\_host\_\_\_\_device\_\_ cudaError_t cudaGetDevice(int* device), \_\_host\_\_ cudaError_t cudaSetDevice(int dev)
```C
int dev;
cudaGetDevice(&dev);
cudaSetDevice(dev);
```
- <i>cudaGetDevice()</i> returns the device on which the active host thread executes the device code into <i>*device</i>.
- <i>cudaSetDevice()</i> set device to be used for GPU executions.

### 2. \_\_host\_\_ cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop)
```C
cudaDeviceProp prop;
memset(&prop, 0, sizeof(cudaDeviceProp));
prop.major = 1;
prop.minor = 3;
cudaChooseDevice(&dev, &prop);
```
- <i>cudaChooseDevice()</i> is the function that returns the closest compute-device to values of <i>*prop</i> into <i>*device</i> by CUDA Runtime.
- We can find the appropriate device to our needs without access all devices by loop.

<br/><br/>

# [05_Parallel programming by CUDA_Vector Sum](https://github.com/unsik6/CUDA_BY_EXAMPLE/blob/main/05_Parallel%20programming%20by%20CUDA_Vector%20Sum.cu)

- keywords: device function, parallel programming, vector sum
 
<br/>

```C
// Single-Core Vector Sum
void add(int* a, int* b, int* c) {
	int tid = 0;	// 0th CPU
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid++;		// only one CPU -> only one increasing
	}
}
```

&nbsp;&nbsp;The code above is 'vector sum' function running in only CPU. And if the CPU has dual core, we can use two core to sum vectors using like the code below.

```C
// the part of code, Dual-Core Vector Sum
void add(int* a, int* b, int* c) {
	int tid = 0;  // 0th CPU // other core initialize 'tid' to 1
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += 2;		// only one CPU -> only one increasing
	}
}
```

&nbsp;&nbsp; But you know that the code above is not enough. You have to design the function <i>add</i> run with kernel. And the function must run in multi-process or multi-thread. Moreover, you have to control the race condtion because scheduling is a non-deterministic part to programmers, and you also have to control the deadlock because the shared memory is two, three arrays <i>a</i>, <i>b</i> and <i>c</i>.<br/>

<p id = "05PP_VS"></p>

### Device function
&nbsp;&nbsp;The code below is the function 'add' written as a kernel functon.

```C
// Kernel function of Vector Sum using 1 grid that has N blocks
__global__ void add(int* a, int* b, int* c) {
	int tid = blockIdx.x;	// what index this block has
	
	// compute the data of this index, if the block is allocated.
	if(tid < N) c[tid] = a[tid] + b[tid];
}

int main(void) {
	///...///
	add <<<N, 1 >>> (dev_a, dev_b, dev_c);
	///...///
}
```

<i><b>\_\_global\_\__kernel<<<the number of blocks, the number of threads per block>>>()</b></i>
&nbsp;&nbsp;<u>The first parameter in three angle brackets is how many blocks will be used.</u> Each block within the grid can be identified by a one-dimensional, two-dimensional, or three-dimensional unique index accessible within the kernel through the built-in blockIdx variable. So, in the kernel function <i>add</i> above, we use <i>blockIdx.x</i>.  And <u>The second parameter is how much threads are in one block.</u><br/>
&nbsp;&nbsp;All blocks run in prarllel. So, we can know that their are <i>N add</i> functions are running in each blocks.
</br></br>
![Grid of Thread Blocks.](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/grid-of-thread-blocks.png)
<center>Grid of Thread Blocks, CUDA Toolkit</center>
<br/>

### Point of Caution
1. You can see the <i>if</i> clause in the kernel function <i>add</i> above. It checks the block called is allocated to run the function. Without that, we may access the memory not allocated - wrong memory.
2. You have to consider the attributes about grids, blocks, thread and memory of <i>cudaDevicProp</i>. You never make the dimesion of block more than <i>cudaDeviceProp.totalConstMem</i>.

<br/>

### The whole process
&nbsp;&nbsp;It is not over that we change the host function to the device function. In some case, we have to copy and paste between host and devices. <br/>
&nbsp;&nbsp;If you construct and fill the arrays <i>a</i> and <i>b</i> in CPU, then


1. Allocate device memory for <i>a</i>, <i>b</i> and <i>c</i>.
2. Copy the data of <i>a</i> and <i>b</i> from host to device.
3. Compute the vector sum by calling device function <i>add</i>.
4. Copy the result data, <i>c</i>, from device to host.






