#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>

__global__ void funcKernel(float* d_in, float *d_out, int N) {
    const unsigned int lid = threadIdx.x; // local id inside a block 
    const unsigned int gid = blockIdx.x*blockDim.x + lid; // global id 
    if (gid < N) {
        d_out[gid] = pow((d_in[gid]/(d_in[gid]-2.3)), 3); // do computation
    }
}

int timeval_subtract(struct timeval* result, struct timeval* t2,struct timeval* t1) { 
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;   result->tv_usec = diff % resolution;
    return (diff<0);
}

#define GPU_RUNS 100
int main(int argc, char** argv) {
    unsigned int N = 1000000000;
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

    ////////// PARALLEL EXEC ////////////////
    // execute the kernel and measure time
    // most websites suggest measuring runtime with cuda events, however we saw the 
    // example with the gettimeofday() function during exercises, so I went with the later!
    unsigned long int elapsed; 
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    // cudaEventRecord(startGPU, stream);
    
    for (int i = 0; i < GPU_RUNS; i++) {
        funcKernel<<< num_blocks, block_size>>>(d_in, d_out, N); // execute kernel
    } cudaThreadSynchronize(); // wait for every thread to finish

    //cudaEventRecord(stopGPU, stream);        // mark end of kernel execution
    //cudaEventSynchronize(stopGPU);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;
    printf("GPU execution took %fms)\n",elapsed/1000.0);

    /////////////////////////////////////////

    // copy result from device to host
    cudaMemcpy(gpu_res, d_out, mem_size, cudaMemcpyDeviceToHost);

    ////////// SEQUENTIAL EXEC //////////////

    unsigned long int elapsed_seq; 
    struct timeval t_start_seq, t_end_seq, t_diff_seq;
    gettimeofday(&t_start_seq, NULL);

    // cudaEventRecord(startGPU, stream);

    for (int i = 0; i < GPU_RUNS; i++) {
        // compute on cpu
        for (int i = 1; i <= N; ++i) {
            cpu_res[i-1] = pow((i/(i-2.3)), 3); // do computation
        }
    }
    //cudaEventRecord(stopGPU, stream); // mark end of kernel execution
    //cudaEventSynchronize(stopGPU);

    gettimeofday(&t_end_seq, NULL);
    timeval_subtract(&t_diff_seq, &t_end_seq, &t_start_seq);
    elapsed_seq = (t_diff_seq.tv_sec*1e6+t_diff_seq.tv_usec) / GPU_RUNS;
    printf("CPU execution took %fms)\n",elapsed_seq/1000.0);

    /////////////////////////////////////////

    // check validty of results
    bool valid = true;
    for (int i = 0; i < N; ++i) {
        if (!(fabs(cpu_res[i] - gpu_res[i]) < 0.00000000001)) { // threshold is beyond floating point precision, so choosing a smaller one would not make a big difference!
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
