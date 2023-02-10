


# CUDA_BY_EXAMPLE

- The text book, Jason Sanders, Edward Kandrot, 'CUDA by Example: An Introduction to General-Purpose GPU Programming<sup>1st</sup>', 2011 
- The major reference: [NVIDA CUDA Toolkit Documentation v11.7.0](https://docs.nvidia.com/cuda/)

- The environment
	> CUDA version: CUDA 11.7 <br/>
	> OS: Window 10 64-bits <br/>
	> GPU: NVIDIA GeForce GTX 1060 6GB <br/>

Unfortunately, All comments or descriptions of source codes are written by korean to improve the efficiency of studying.

# Contents
 [00_start from Jun 27, 2022](#00_start-from-June-27-2022) <br/>
 [01_Host Code and Device Code](#01_Host-Code-and-Device-Code) <br/>
 [02_Allocating and Using Device Memory](#02_Allocating-and-Using-Device-Memory) <br/>
 [03_Query of Device Properties](#03_Query-of-Device-Properties) <br/>
 [04_Use Device Properties_Find appropriate CUDA device](#04_Use-Device-Properties_Find-appropriate-CUDA-device) <br/>
 [05_Parallel programming by CUDA: Vector Sum](#05_Parallel-programming-by-CUDA_Vector-Sum) <br/>
 [06_Parallel programming by CUDA_Julia Set](#06_Parallel-programming-by-CUDA_Julia-Set) <br/>
 [07_Multi thread by CUDA_Vector Sum](#07_Multi-thread-by-CUDA_Vector-Sum) <br/>
 [08_Multi thread by CUDA_Ripple](#08_Multi-thread-by-CUDA_Ripple) <br/>
 [09_Shared Memory_Vector dot product](#09_Shared-Memory_Vector-dot-product) <br/>
 [10_Constant Memory and Time recording](#10_Constant-Memory-and-Time-recording)<br/>

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

# [05_Parallel programming by CUDA_Vector Sum](https://github.com/unsik6/CUDA_BY_EXAMPLE/blob/main/Source%20codes%20of%20Examples/05_Parallel%20programming%20by%20CUDA_Vector%20Sum.cu)

- keywords: device function, grid, block, thread, parallel programming, vector sum
 
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

&nbsp;&nbsp; But you know that the code above is not enough. You have to design the function <i>add</i> run with kernel. And the function must run in multi-process or multi-thread. Moreover, you have to control the race condtion because scheduling is a non-deterministic part to programmers, and you also have to control the deadlock because the shared memory is three arrays <i>a</i>, <i>b</i> and <i>c</i>. (Actually, there is no shared memory, since processors access memories.) <br/>

<p id = "05PP_VS"></p>

### 1. Device function
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
&nbsp;&nbsp;All blocks run in prarllel. So, we can know that there are <i>N add</i> functions are running in each blocks.
</br></br>
Fig 1.Grid of Thread Blocks, CUDA Toolkit

![image](https://user-images.githubusercontent.com/80208196/218024368-33f121a2-3602-4b87-871f-0dad08d12abc.png)

<br/>

<p id = "05PP_VS_2"></p>

### 2. Point of Caution
1. You can see the <i>if</i> clause in the kernel function <i>add</i> above. It checks the block called is allocated to run the function. Without that, we may access the memory not allocated - wrong memory.
2. You have to consider the attributes about grids, blocks, thread and memory of <i>cudaDevicProp</i>. You never make the dimesion of block more than <i>cudaDeviceProp.maxGridSize</i>.


### 3. The whole process
&nbsp;&nbsp;It is not over that we change the host function to the device function. In some case, we have to copy and paste between host and devices. <br/>
&nbsp;&nbsp;If you construct and fill the arrays <i>a</i> and <i>b</i> in CPU, then


1. Allocate device memory for <i>a</i>, <i>b</i> and <i>c</i>.
2. Copy the data of <i>a</i> and <i>b</i> from host to device.
3. Compute the vector sum by calling device function <i>add</i>.
4. Copy the result data, <i>c</i>, from device to host.

<br/><br/>

# [06_Parallel programming by CUDA_Julia Set](https://github.com/unsik6/CUDA_BY_EXAMPLE/blob/main/06_Parallel%20programming%20by%20CUDA_Julia%20Set.cu)

- keywords: device function, grid, block, thread, parallel programming, Julia Set
<br/>

Fig 2. The output of the Julia Set GPU application example

![Julia Set Example](https://user-images.githubusercontent.com/80208196/180773059-f3d2189a-9835-43ab-bb0b-7885e179c01a.PNG)

<center>The output of Julia Set Example</center>

&nbsp;&nbsp; The difference between [Julia Set](https://en.wikipedia.org/wiki/Julia_set) CPU application and Julia Set GPU application is same with the previous example, Vector Sum. So, we are just talking about passing the number of block by grid. <br/>

```C
__global__ void kernel(unsigned char* ptr) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;
	
	int juliaValue = julia(x, y);
	ptr[offset * 4 + 0] = 255 * juliaValue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

#define DIM 1000

int main(void) {
	///...///
	dim3 grid(DIM, DIM);
	kernel <<< grid, 1 >>> (dev_bitmap);
	///...///
}
```

<br/>
&nbsp;&nbsp;You can see that the type of first parameter of <i>kernel</i> device function is <i>dim3</i>. <u><i>dim3</i> is an integer vector type based on <i>uint3</i> that is used to sepcify dimensions.</u> This type is maximum three-dimension. <b>Although the three-dimension grid is not supported, <i>CUDA Runtime</i> expect passing this type.</b> If you don't pass three parameter, the dimension that get no parameter is initialized to 1. <br/>
&nbsp;&nbsp;In the code above, kernel device function runs in two-dimensional grid by passing the <i>dim3</i> type variable initialized by two parameters. So, each block means each position of an image, and you can see that the <i>kernel</i> device function of the each block runs one time for each block position = image position.
<br/>

- girdDim: dim3 type; contains the dimensions of the grid.

<br/><br/>

# [07_Multi thread by CUDA_Vector Sum](https://github.com/unsik6/CUDA_BY_EXAMPLE/blob/main/07_MultiThread%20by%20CUDA_Vector%20Sum.cu)

- keywords: multi-thread, multi-core, device function, grid, block, thread, parallel programming, vector sum
<br/>

### 1. Limit: The number of threads per block
&nbsp;&nbsp;In the previous chapter, we use multi-block for parallel programming([05_Parallel programming by CUDA: Vector Sum](#05_Parallel-programming-by-CUDA_Vector-Sum)). However, we can use multi-thread for same thing. The pros and cons will be discussed later.

``` C
// Case 1: One block & N threads
__global__ void add(int* a, int* b, int* c)
{
	int tid = threadIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}
///...///
add << <1, N >> > (dev_a, dev_b, dev_c);
```
&nbsp;&nbsp;We know that the second parameter is how much threads are in one block. The code above runs in one block and <i>N</i> threads. But, there is a limit of the number of threads per block and we can find that using a query about <i>cudaDeviceProp.maxThreadsPerBlock</i>. If the number of threads per block we use is greater than the limit, the code can't run.

```C
// Case 2: Multi blocks & Multi threads - Consider the limit of the number of threads
__global__ void add(int* a, int* b, int* c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}
///...///
add << <(N+127)/128, 128 >> > (dev_a, dev_b, dev_c);
```
&nbsp;&nbsp;The code above runs in (<i>N</i>+127)/128 blocks and 128 threads per block. Actually, the limit of the number of threads is smaller than the limit of size of grid. So, we can fix the number of thread per block and use more blocks. To use appropriate number of blocks, we have to round up the result of division. Else, the total number of threads is less than the number we need.
> Example: we need 130 threads and use 128 threads per block. <br/>
	> (1) N / 128 = 1 => we use one block. So, total number of threads we can use is 128.<br/>
	> (2) (N+127)/128 => we use two blocks. So, total number of threads we can use is 256.<br/>

&nbsp;&nbsp;In this way, we need to compute an index of each thread more complicatedly. <b><i>blockDim</i> is a constant variable. <i>blockDim</i> stores the number of used threads of each block. The supported maximum dimension of grid is two, but The maximum dimension of block is three.</b> In this example, since we give the number of threads per block as one dimension, we can consider the relation between blocks and threads as a matrix. Consider indices of blocks as indices of row of a matrix and consider indices of threads as indices of columns of a matrix.
<br/>

### 2. Limit: The number of threads per block & The grid size
&nbsp;&nbsp; But we already know there is <a href = "#05PP_VS_2"> the limit of the grid size</a>. If <i>N</i>/128 is greater than the dimension of grid, the code above doesn't work.
```C
__global__ void add(int* a, int* b, int* c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	while (tid < N)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}
///...///
add << <128, 128 >> > (dev_a, dev_b, dev_c);
```
&nbsp;&nbsp;We can use the threads of GPU as cores, like the code above. In this case, we use only 128 blocks and 128 threads per block. This constant numbers can be changed if the changed numbers do not go over the limit already discussed. So, what we have to consider is only whether the space needed for arrays storing a vector is less than the constant memory of a device(GPU), <i>cudaDeviceProp.totalConstMem</i>.

<br/><br/>

# [08_Multi thread by CUDA_Ripple](https://github.com/unsik6/CUDA_BY_EXAMPLE/blob/main/08_MultiThread%20by%20CUDA_Ripple.cu)

- keywords: multi-thread, grid, block, thread, parallel programming, ripple animation
<br/>

Fig 3. The output of the ripple animation example

![ripple](https://user-images.githubusercontent.com/80208196/184888142-7d030d53-ffff-4957-87c5-5f7d6b54b70a.gif)


&nbsp;&nbsp; In this chapter, we use the given library, <i>cpu_anim</i>, which processes the operations for animations. The main process is simple and equals what we already studyed. One core practices of this chapter is computing an image by passessing the number of blocks and the number of threads per block as two dimensional parameter whose type is <i>dim3</i>, and another one is what we need to consider when we make an animation.
```C
struct DataBlock{
	unsigned char *dev_bitmap;
	CPUAnimBitmap *bitmap;
}
///...///
int main(void){
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));
	bitmap.anim_and_exit((void(*)(void*, int)) generate_frame, (void(*)(void*)) cleanup);
}
```
&nbsp;&nbsp; <i>DIM</i> is the width(height), by pixel, of an image. <b>First, allocate the linear (unsigned char array) device memory as many as the two dimensional image size.</b> <i>bitmap.anim_and_exit</i> is just the function calling <i>generate_frame</i> once per frame by the class of the given library. (<i>cleanup</i> is just the function to deallocate the device memory.)

### 1. Two dimensional blocks and threads

```C
void generate_frame(DataBlock *d, int ticks) {
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel<<<blocks, threads>>> (d->dev_bitmap, ticks);
	cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHOst);
```
&nbsp;&nbsp;The function <i>generate_frame</i> is called once per frame. <i>kernel</i> is just the function compute the color of each pixel. Focus on two variables, <i>blocks</i> and <i>threads</i>. We pass 16 by 16 threads per block and <i>DIM</i>/16 by <i>DIM</i>/16 blocks to the function <i>kernel</i>. It means total number of threads is <i>DIM</i> * <i>DIM</i>. It is easy to think of the hierarchical structure of this memory. Each thread will compute the information of each pixel.

### 2. Compute the position of a thread

```C
__global__ void kernel (unsigned char *ptr, int ticks) {
	// compute the position of pixel computed, using threadIdx and blockIdx.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	///Compute the color of the pixel.///
}
```
&nbsp;&nbsp;The part of the function <i>kernel</i> for computing a color of a pixel is left out, since the part is just for making the ripple. Focus on three variables, <i>x</i>, <i>y</i> and <i>offset</i>. <i>x</i> and <i>y</i> are the real position of the thread which runs <i>kernel</i> in  a matrix about whole threads. And <i>offset</i> is the linear offset of the position. Since we allocated the device memory for an image as linear array, we need to compute the linear offset.

> <b>Q. What is the parameter <i>ticks</i>?</b>
> 
> <b>A.</b> <i>ticks</i> is time. To compute the exact color of each pixel, the device needs the information of real time of animation,

<br/><br/>

# [09_Shared Memory_Vector dot product](https://github.com/unsik6/CUDA_BY_EXAMPLE/blob/main/09_Shared%20Memory_Vector%20dot%20product.cu)

- keywords: multi-thread, shared memory, race condition, synchronization,  grid, block, thread, parallel programming, reduction, vector dot product, __syncthreads
<br/>

### 1. Shared memory
```C
__global__ void dot(float *a, float *b, float *c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIdx = threadIdx.x;
	// compute subsum of dot product
	float temp = 0;
	while(tid < N) {
		temp += a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIdx] = temp;
	__syncthreads();

	/// reduction ///
}
```
&nbsp;&nbsp; In this example, we compute dot product(inner product) of vectors of a long length. How to construct <i>grid</i>, <i>blocks</i> and <i>threads</i> equals what we study at the previous chapter. So, the core part of this chapter is <u><b>shared memory</b></u>. <i>CUDA C</i> compiler manages the variables declared with <i><b>\_\_shared\_\_</b></i> specially. The <i>\_\_shared\_\_</i> variables is the memory shared among theards in the same block. The latency to this memories is more shorter than others, since the memories reside physically in GPU, not DRAM. <br/>
&nbsp;&nbsp;In this example, each thread in a same block uses one element of the array <i>cache</i>, shared memory. Threads compute the results of performing the dot product for each element of the vector assigned to them. And they sum all results, and store the partial scalar sum into <i>cache</i>. <br/>
&nbsp;&nbsp;After then, the function <i>dot</i> will sum all partial scalar sum to compute the result of dot product. (reduction phase) But we have to assure that all partial scalar sum is already compute. For that, we use the function <b><u><i>\_\_syncthreads</i></u></b>. This function make the threads, which already complete all computation, wait until all threads of their block finish the computation. To be exact, The codes after <i>\_\_syncthreads</i> only run after all threads of a same block run all code before <i>\_\_syncthreads</i>.<br/>

### 2. Reduction

&nbsp;&nbsp;Reduction is popular computing method in parallel programming. In this phase, settle all result of a lot of computation ended in parallel. In this example, if we sum all partial scalar sum by naive approach, we need the time prportional to the number of threads. But reduction phase also use parallel programming.
```C
__global__ void dot(float *a, float *b, float *c) {
	/// compute all parts of process in parallel ///
	int i = blockDim.x;
	while(i != 0) {
		if(cacheIdx < i)
			cache[cacheIdx] += cache[cacheIdx + i];
		__syncthreads();
		i /= 2;
	}
	if(cacheIdx == 0)
		c[blockIdx.x] = cache[0];
}
```
&nbsp;&nbsp;The code above runs in <i>log(the number of threads per block)</i> time. Half threads of a block sum two partial scalar sum in the shared arrya <i>cache</i>, And this process iterate until only one value - the sum of partial scalar sums in a block. (The number of threads per block has to be the power of two.) Between stages, the threads using array <i>cache</i> have to be synchronized. After end of this phase, store the result into a global variable with host.

### 3. CAUTION: Wrong optimization - deadlock

&nbsp;&nbsp;You may think <i>\_\_syncthreads</i> is called only when summing elements of <i>cache</i>, and may revise the code above like one below.
```C
int i = blockDim.x;
while(i != 0) {
	if(cacheIdx < i) {
		cache[cacheIdx] += cache[cacheIdx + i];
		__syncthreads();
	}
	i /= 2;
```
&nbsp;&nbsp;But this code causes deadlock. Since all threads in a same block run <i>\_\_syncthreads</i> to run the codee after <i>\_\_syncthreads</i>, and the threads, whose index is out of range for reduction but used when computing each element of vector, never run the function, the program will be stop.

<br/><br/>

# [10_Constant Memory and Time recording](https://github.com/unsik6/CUDA_BY_EXAMPLE/tree/main/Source%20codes%20of%20Examples)

- keywords: constant memory, cudaEvent, time recording, multi-thread, grid, block, thread, parallel programming, ray tracing
<br/>

Fig 4. The output of the ray tracing example
![ray_tracing_example](https://user-images.githubusercontent.com/80208196/186683802-e3a1b5a9-8fc7-4b56-b2f5-355a7e7c09ad.PNG)

### 1. Constant memory

```C
__constant__ Sphere s[SPHERES];
```

&nbsp;&nbsp;We can declare constant memory using keyword <i>\_\_constant\_\_</i>. Hardwares of NVIDIA grant 64kb constant memory. Constant memory is only read. If we use constant memory, memory bandwidth can be more less. There is two reaseon.

> <b>1. If threads call a constant memory, the number of calls can be reduced to 1/16.</b><br/>
	&nbsp;&nbsp;When using constant memory, Hardewares of NVIDIA will broadcast the call to all threads of the half-warp of the threads which really call the memory. Since a half-warp is consist of 16 threads, other threads which need to call the memory will not call the same constant memory again. <u>It makes run time more less. However, since accessing of constant memory is executed sequentially. run time can more greater if threads of the same half-warp calls different memories. The process that may be performed in parallel is serialized</u>. <br/><br/>
> <b>2. Constant memory is taken from a cache of GPU.</b><br/>
	&nbsp;&nbsp;Since constant memory is never revised, GPU will caching constant memory enthusiastically. So, the constant memory will be hit more. It reduces the number of using memory bus.

### 2. CUDA Event
&nbsp;&nbsp;When we decide what program, code or algorithm is more better than others, we use some measurement, such as time. There are some API about time event in CUDA C. We can use libraries of C or timer of OS, but it is not sure that measured time is precise. There are many variables such as scheduling of OS, effectiveness of CPU timer. The most important reason of this impreciseness is that host may compute time without synchronization with GPU kerne. So, we will use API of events of CUDA. Events of CUDA is the time stamp of GPU, and it records time when programmer specify.
```C
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);

///...run...///

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);
```
&nbsp;&nbsp;Generally, a recording time process is 'create cudaEvent -> record -> destroy cudaEvent'. It is similiar with allocating and using memories. <i><b>cudaEvent_t</b></i> is similar with a marker.
> \_\_host\_\_ cudaError_t cudaEventCreate ( cudaEvent_t* event ): create an event obect.<br/>
> \_\_host\_\_\_\_device\_\_ cudaError_t cudaEventRecord ( cudaEvent_t event, cudaStream_t stream = 0 ): record an event.<br/>
> \_\_host\_\_ cudaError_t cudaEventSynchronize ( cudaEvent_t event ): waits for an event to complete.<br/>
> \_\_host\_\_ cudaError_t cudaEventElapsedTime ( float *ms, cudaEvent_t start, cudaEvent_t end ): computes the elapsed time between events and store the time(ms) into the first parameter.<br/>
