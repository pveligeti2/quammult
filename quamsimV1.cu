#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>


// The cuda kernel
__global__ void quamsim_kernel(const float *qstates,const float *qbit_gate, float *q_out, int n_size,int q_index ) {


  int i = blockIdx.x* blockDim.x + threadIdx.x;

  int i_opp = i ^ (1<<q_index);

  if(i<n_size){
    if((i_opp & (1<<q_index))){
      q_out[i] = qbit_gate[0] * qstates[i] + qbit_gate[1] * qstates[i_opp];
      q_out[i_opp] = qbit_gate[2] * qstates[i] + qbit_gate[3] * qstates[i_opp];
    }
  }
}


int main(int argc, char *argv[]) {

  // Read the inputs from command line

  char *trace_file;
  trace_file = argv[1];

  std::ifstream file(trace_file);

  float gate[4];

  for(int i = 0; i<4; i++){
      
          file >> gate[i];
      
  }

  std::vector<float> states;
  
  float instate;
  while(file>>instate){
      states.push_back(instate);
  }

  int t;
  t = states.back();
  states.pop_back();

  int n = states.size();

  size_t size = n* sizeof(float);
  size_t size_gate = 4 * sizeof(float);
  // Allocate/move data using cudaMalloc and cudaMemCpy

  float *host_states = (float*) malloc(size);
  float *host_gate = (float*)malloc(size_gate);

  float *host_out = (float*)malloc(size);

  for (int i = 0; i < n; ++i)
  {
      host_states[i] = states[i];
  }

  for (int i = 0; i < 4; i++) 
  {
      host_gate[i]= gate[i];
  }

  //allocating the memory in GPU
  float *d_states, *d_gate, *d_out;

  cudaMalloc((void**)&d_states, size);
  cudaMalloc((void**)&d_gate, size_gate);
  cudaMalloc((void**)&d_out, size);

  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  
  cudaMemcpy(d_states, host_states, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gate, host_gate, size_gate, cudaMemcpyHostToDevice);



  int threadsPerBlock = 256;
  int blocksPerGrid = (n+threadsPerBlock-1/threadsPerBlock);

  // Launch the kernel
  cudaEventRecord(start);
  quamsim_kernel<<<blocksPerGrid, threadsPerBlock >>>(d_states,d_gate,d_out,n, t);
  cudaEventRecord(stop);
  // Print the output

  // Clean up the memory
  cudaMemcpy(host_states, d_states, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_gate, d_gate, size_gate, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_out, d_out, size, cudaMemcpyDeviceToHost);

//printing to be written

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout<< milliseconds << std::endl;
  for(int i = 0; i<n; i++){
    std::cout<< std::fixed << std::setprecision(3) << host_out[i] << std::endl;
  }
  cudaFree(d_states);
  cudaFree(d_gate);
  cudaFree(d_out);

  free(host_states);
  free(host_gate);
  free(host_out);


  
  
  printf("Done\n");
  return 0;
}
