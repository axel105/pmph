#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void funcKernel(float* d_in, float *d_out, int N) {
    const unsigned int lid = threadIdx.x; // local id inside a block 
    const unsigned int gid = blockIdx.x*blockDim.x + lid; // global id 
    if (gid < N) {
        d_out[gid] = pow((d_in[gid]/(d_in[gid]-2.3)), 3); // do computation
    }
}

int main(int argc, char** argv) {
    unsigned int N = 753411;
    unsigned int mem_size = N*sizeof(float);
    unsigned int block_size  = 1024;
    unsigned int num_blocks  = ((N + (block_size - 1)) / block_size);

    // allocate host memory
    float* h_in  = (float*) malloc(mem_size);
    float* cpu_res = (float*) malloc(mem_size);
    float* gpu_res = (float*) malloc(mem_size);
    // initialize the memory
    for(unsigned int i=1; i<=N; ++i){ 
        h_in[i] = (float) i+1;
    }

    // allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    // execute the kernel
    funcKernel<<< num_blocks, block_size>>>(d_in, d_out, N);

    // copy result from device to host
    cudaMemcpy(gpu_res, d_out, mem_size, cudaMemcpyDeviceToHost);

    printf("exectued something:   %.6f\n", gpu_res[3342]);
    
    // clean-up memory
    free(h_in);       
    free(cpu_res);
    free(gpu_res);
    cudaFree(d_in);
    cudaFree(d_out);
}
