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
        h_in[i-1] = (float) i;
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

    // compute on cpu
    for (int i = 1; i <= N; ++i) {
        cpu_res[i-1] = pow((i/(i-2.3)), 3); // do computation
    }

    // check validty of results
    bool valid = true;
    for (int i = 0; i < N; ++i) {
        if (!(fabs(cpu_res[i] - gpu_res[i]) < 0.00000000000001)) {
            valid = false;
            printf("CPU res: %f, GPU res: %f\n", cpu_res[i], gpu_res[i]);
        }
    }

    if (valid) {
        printf("VALID\n");
    } else {
        printf("INVALID\n");
    }
    
    // clean-up memory
    free(h_in);       
    free(cpu_res);
    free(gpu_res);
    cudaFree(d_in);
    cudaFree(d_out);
}
