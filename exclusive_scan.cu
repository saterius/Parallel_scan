#include<stdio.h>
#include<math.h>

#define N 512

__global__ void exclusive_scan(int *d_in) {

	__shared__ int temp_in[N];

	int id = threadIdx.x;
	temp_in[id] = d_in[id];

	__syncthreads();

	unsigned int s = 1;
	//int i = 2*s*(threadIdx.x + 1) - 1;
	for(; s <= N-1; s <<= 1) {
		int i = 2*s*(threadIdx.x + 1) - 1;
		if((i >= s) && (i < N)) {
			int a = temp_in[i];
			int b = temp_in[i-s];
			int c = a + b;
			temp_in[i] = c;
		}
		__syncthreads();
	}

	//d_in[i] = temp_in[i];

	if(threadIdx.x == 0) {
		d_in[N-1] = 0;
		temp_in[N-1] = 0;
	}

	for(s = s/2; s >= 1; s >>= 1) {
		int i = 2*s*(threadIdx.x + 1) - 1;
		if((i <= s) && (i < N)) {
			int r = temp_in[i];
			int l = temp_in[i-s];
			__syncthreads();
			temp_in[i] = l + r;
			temp_in[i-s] = r;
		}
		__syncthreads();
	}

	d_in[id] = temp_in[id];

}

int main()
{
	int h_in[N];
	int h_out[N];

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for(int i=0; i < N; i++)
		h_in[i] = 1;

	int *d_in;
	//int *d_out;

	cudaMalloc((void**) &d_in, N*sizeof(int));
	//cudaMalloc((void**) &d_out, N*sizeof(int));
	cudaMemcpy(d_in, &h_in, N*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaEventRecord(start);

	//Implementing kernel call
	exclusive_scan<<<1, N>>>(d_in);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(&h_out, d_in, N*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++)
		printf("out[%d] = %d\n", i, h_out[i]);

	cudaFree(d_in);
	//cudaFree(d_out);

	printf("Time used: %f milliseconds\n", milliseconds);

	return -1;

}
