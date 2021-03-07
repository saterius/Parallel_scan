#include<stdio.h>
#include<math.h>

#define N 512

__global__ void inclusive_scan(int *d_in) {

	__shared__ int temp_in[N];

	int i = threadIdx.x;
	temp_in[i] = d_in[i];

	__syncthreads();

	for(unsigned int s = 1; s <= N-1; s <<= 1) {
		if((i >= s) && (i < N)) {
			int a = temp_in[i];
			int b = temp_in[i-s];
			int c = a + b;
			temp_in[i] = c;
		}
		__syncthreads();
	}

	d_in[i] = temp_in[i];
}

int main()
{
	int h_in[N];
	int h_out[N];

	for(int i=0; i < N; i++)
		h_in[i] = 1;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *d_in;
	//int *d_out;

	cudaMalloc((void**) &d_in, N*sizeof(int));
	//cudaMalloc((void**) &d_out, N*sizeof(int));
	cudaMemcpy(d_in, &h_in, N*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaEventRecord(start);

	//Implementing kernel call
	inclusive_scan<<<1, N>>>(d_in);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(&h_out, d_in, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	//cudaFree(d_out);

	for(int i=0; i<N; i++)
		printf("out[%d] =  %d\n", i, h_out[i]); 

	printf("Time used: %f milliseconds\n", milliseconds);

	return -1;

}
