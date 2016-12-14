#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>

#include <hdf5.h>
#include <wb.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10
#define BLOCK_SIZE 512

static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32};
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};

static int loadData(float *x, float *y) {
  // Open the data file
  const auto file_id =
      H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset x and y
  const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
  const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

  // Get the dataset x dimensions
  const auto xspace = H5Dget_space(x_id);
  const auto xndims = H5Sget_simple_extent_ndims(xspace);
  assert(xndims == 4);

  hsize_t input_dims[xndims];
  H5Sget_simple_extent_dims(xspace, input_dims, NULL);
  if (input_dims[0] != FLAGS_batch_size) {
    std::cout << "data size does not match batch size specified!\n";
    return 1; // return error
  }
  std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
            << " x " << input_dims[2] << " x " << input_dims[3] << "\n";

  // Read the dataset x and y
  check_success(
      H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(
      H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

  // Close the dataset x and y
  check_success(H5Dclose(x_id));
  check_success(H5Dclose(y_id));

  // Close the file
  check_success(H5Fclose(file_id));

  // return success
  return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
  // Open the model file
  const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset
  const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
  const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
  const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
  const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

  // Read the dataset
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv2));
  check_success(
      H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(
      H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}

// From book chapter Figure 16.4
static void conv_forward_valid(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
  const auto filter_h   = wdims[0];
  const auto filter_w   = wdims[1];
  const auto in_channel = wdims[2];

  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {
      for (const auto w : range(0, ydims[2])) {
        for (const auto h : range(0, ydims[1])) {
          for (const auto p : range(0, filter_h)) {
            for (const auto q : range(0, filter_w)) {
              for (const auto c : range(0, in_channel)) {
                const auto yoffset =
                    ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
                const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
                                     (h + p) * xdims[2] * xdims[3] +
                                     (w + q) * xdims[3] + c;
                const auto woffset = p * wdims[1] * wdims[2] * wdims[3] +
                                     q * wdims[2] * wdims[3] + c * wdims[3] + m;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Recified linear unit 4d
static void relu4(float *X, const int xdims[4]) {
  for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

__global__ void relu4_kernel(float *X, const int xdims[4]){
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  if(row < xdims[0] * xdims[1] * xdims[2] * xdims[3])
    X[row] = (X[row] < 0) ? 0 : X[row];
}

// Recified linear unit 2d
static void relu2(float *X, const int xdims[2]) {
  for (const auto i : range(0, xdims[0] * xdims[1])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// From book chapter Figure 16.5
static void average_pool(const float *X, const int xdims[4],
                         const int pool_size, float *Y, const int ydims[4]) {
  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {
      for (const auto w : range(0, ydims[2])) {
        for (const auto h : range(0, ydims[1])) {
          for (const auto p : range(0, pool_size)) {
            for (const auto q : range(0, pool_size)) {
              const auto yoffset =
                  ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
              const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
                                   (pool_size * h + p) * xdims[2] * xdims[3] +
                                   (pool_size * w + q) * xdims[3] + m;
              Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
            }
          }
        }
      }
    }
  }
}

static void fully_forward(const float *X, const int xdims[2], float *W,
                          const int wdims[2], float *Y, const int ydims[2]) {
  for (const auto i : range(0, xdims[0])) {
    for (const auto j : range(0, wdims[1])) {
      float sum = 0;
      for (const auto k : range(0, xdims[1])) {
        sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
      }
      Y[i * wdims[1] + j] = sum;
    }
  }
}

// Choose the guess with largest score
static void argmax(const float *X, const int xdims[2], int *Y) {
  
  for (const auto i : range(0, xdims[0])) {
    auto max_idx = 0;
    auto max     = X[i * xdims[1]];
    for (const auto j : range(0, xdims[1])) {
      const auto elem = X[(i * xdims[1]) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
}

__global__ void argmax_kernel(const float *input, const int fdims[2], int *output) {

  int i= threadIdx.x;
  auto max_idx = 0;
  auto max     = input[i * fdims[1]];
  for (const auto j : range(0, fdims[1])) {
    const auto elem = input[(i * fdims[1]) + j];
    if (elem > max) {
      max_idx = j;
      max     = elem;
    }
  }
  output[blockIdx.x*1024+i] = max_idx;
  
  // const auto fdims0_size = fdims[0];
  //correct start
  // __shared__ int max_idx[16];
  // __shared__ float max[16];

  // int t_x = threadIdx.x;
  // int b_x = blockIdx.x;
  // int dim_x = blockDim.x;
  // int start_index = b_x*10;

  // // initializing the two parts
  // if(start_index+t_x < start_index+10){
  //   // output[1] = 1;
  //   max_idx[t_x] = t_x;
  //   max[t_x] = input[start_index+t_x];
  // }
  // else{
  //   if(t_x<16){
  //     // output[2] = 2;
  //   max_idx[t_x] = t_x;
  //   max[t_x] = 0;
  //   }
    
  // } end

  
  // if(start_index+dim_x+t_x < fdims[0]*fdims[1]){
  //   // output[3] = 3;
  //   max_idx[dim_x+t_x] = dim_x+t_x;
  //   max[dim_x+t_x] = input[start_index+dim_x+t_x];
  // }
  // else{
  //   // output[4] = 4;
  //   max_idx[dim_x+t_x] = 0;
  //   max[dim_x+t_x] = input[start_index];
  // }

  // // sanity check
  // int n = sizeof(max_idx)/sizeof(max_idx[0]);
  // for(int i = 0; i < n; i++){
  //   output[i] = max_idx[i];
  // }
  // output[5] = 5;

  // comparison
  // for(int idx =; idx >= 1; idx/=2){
  //   __syncthreads();
  //   if(t_x < idx){
  //     if(max[t_x] < max[t_x+idx]){
  //       max_idx[t_x] = max_idx[t_x+idx];
  //       max[t_x] = max[t_x+idx];
  //     }
  //   }
  //   output[b_x] = max_idx[0];
  // }

}

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  // conv layer
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  auto a = zeros<float>(adims);
  conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);

  /// relu layer
  //relu4(a, adims);
  float *device_a;
  int *device_adims;
  int device_a_size = adims[0]*adims[1]*adims[2]*adims[3];

  cudaMalloc((void **)&device_adims, sizeof(int)*4);
  cudaMemcpy(device_adims, adims, sizeof(int)*4, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_a, sizeof(float)* device_a_size);
  cudaMemcpy(device_a, a, sizeof(float)*device_a_size, cudaMemcpyHostToDevice);

  dim3 DimGrid(ceil(device_a_size/256), 1, 1);
  dim3 DimBlock(256, 1, 1);
  relu4_kernel<<<DimGrid, DimBlock>>>(device_a, device_adims);

  cudaMemcpy(device_adims, adims, sizeof(int)*4, cudaMemcpyDeviceToHost);
  cudaMemcpy(device_a, a, sizeof(float)*device_a_size, cudaMemcpyDeviceToHost);

  // average pooling
  const int pool_size = 2;
  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                       adims[3]};
  auto b = zeros<float>(bdims);
  average_pool(a, adims, pool_size, b, bdims);

  // conv layer
  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
                       (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  auto c = zeros<float>(cdims);
  conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

  // relu
  relu4(c, cdims);

  // average pooling
  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                       cdims[3]};
  auto d = zeros<float>(ddims);
  average_pool(c, cdims, pool_size, d, ddims);

  // reshape
  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

  // matrix multiplication
  const int edims[] = {ddims[0], fc1dims[1]};
  auto e            = zeros<float>(edims);
  fully_forward(d, ddims2, fc1, fc1dims, e, edims);

  // relu
  relu2(e, edims);

  // matrix multiplication
  const int fdims[] = {edims[0], fc2dims[1]};
  auto f            = zeros<float>(fdims);
  fully_forward(e, edims, fc2, fc2dims, f, fdims);

  std::cout << "fdims[0]: " << fdims[0] << "\n";
  std::cout << "fdims[1]: " << fdims[1] << "\n";

  float* deviceF;
  int* deviceOUT;
  int* deviceFDIMS;

  int numInputElements = fdims[0]*fdims[1];  // number of elements in the input list
  int numOutputElements = fdims[0];          // number of elements in the output list

  cudaMalloc((void**)&deviceF,numInputElements*sizeof(float));
  cudaMalloc((void**)&deviceOUT,numOutputElements*sizeof(int));
  cudaMalloc((void**)&deviceFDIMS, 2*sizeof(int));

  cudaMemcpy(deviceF, f, numInputElements*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceFDIMS, fdims, 2*sizeof(int), cudaMemcpyHostToDevice);

  // for testing
  // float* g1;
  // int* g2;

  // g1=allocate<float>
  // cudaMemcpy(g1, deviceF, numInputElements*sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(g2, deviceFDIMS, 2*sizeof(int), cudaMemcpyHostToDevice);
  // std::cout << "g1: \n";
  // for (int i = 0; i < numInputElements; i++){
  //   std::cout << g1[i] << "\n";
  // }
  //

  // int thread_num = (numInputElements-1)/2+1;

  // dim3 dimGrid(10, 1, 1);
  // dim3 dimBlock(32, 1, 1);

  dim3 dimGrid(ceil(fdims[0]/1024.0), 1, 1);
  dim3 dimBlock(1024, 1, 1);
   
  argmax_kernel<<<dimGrid,dimBlock>>>(deviceF,deviceFDIMS,deviceOUT);

  // cudaDeviceSynchronize();
  cudaMemcpy(out, deviceOUT, numOutputElements * sizeof(int), cudaMemcpyDeviceToHost);
  
  // std::cout << "out: \n";
  // for (int i = 0; i < sizeof(out)/sizeof(out[0]); i++){
  //   std::cout << out[i] << "\n";
  // }

  cudaFree(deviceF);
  cudaFree(deviceOUT);
  cudaFree(deviceFDIMS);

  // argmax(f, fdims, out);

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  delete[] f;
}

int main(int argc, char **argv) {
  int deviceCount;
  //wbArg_read(argc, argv);
  cudaGetDeviceCount(&deviceCount);

  //wbTime_start(GPU, "Getting GPU Data."); //@@ start a timer
  for (int dev = 0; dev < deviceCount; dev++) {
    //std::cout << "Device Count: " << dev;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cout << "Device " << dev << " name: " << deviceProp.name << "\n";
    if (dev == 0) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        std::cout << "No CUDA GPU has been detected \n";
        return -1;
      } else if (deviceCount == 1) {
        //@@ WbLog is a provided logging API (similar to Log4J).
        //@@ The logging function wbLog takes a level which is either
        //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or TRACE and a
        //@@ message to be printed.
        std::cout << "There is 1 device supporting CUDA" << "\n";
      } else {
        std::cout << "There are " << deviceCount << " devices supporting CUDA" << "\n";
      }
    }

    std::cout << "Device " << dev << " name: " << deviceProp.name << "\n";
    std::cout << " Computational Capabilities: " << deviceProp.major << "." << deviceProp.minor << "\n";
    std::cout << " Maximum global memory size: " << deviceProp.totalGlobalMem << "\n";
    std::cout << " Maximum constant memory size: " << deviceProp.totalConstMem << "\n";
    std::cout << " Maximum shared memory size per block: " << deviceProp.sharedMemPerBlock << "\n";
    std::cout << " Maximum block dimensions: " << deviceProp.maxThreadsDim[0] << " x " << deviceProp.maxThreadsDim[1]
        << " x " << deviceProp.maxThreadsDim[2] << "\n";
    std::cout << " Maximum grid dimensions: " << deviceProp.maxGridSize[0] <<
          " x " << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << "\n";
    std::cout << " Warp size: " << deviceProp.warpSize << "\n";
  }

  //wbTime_stop(GPU, "Getting GPU Data."); //@@ stop the timer

  if (argc != 3 && argc != 4) {
    std::cerr << "\n"
              << "This program performs the forward opertion step for "
                 "Convolutional Neural Network(CNN).  "
                 "Sample usage: \n"
              << argv[0]
              << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
    return -1;
  }
  FLAGS_testdata = std::string(argv[1]);
  FLAGS_model    = std::string(argv[2]);
  if (argc == 3) {
    const std::map<std::string, int> default_batch_sizes{
        {"../data/test2.hdf5", 2},
        {"../data/test10.hdf5", 10},
        {"../data/test100.hdf5", 100},
        {"../data/testfull.hdf5", 10000}};
    const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
    if (batch_size_in_map == default_batch_sizes.end()) {
      std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
      return -1;
    }
    FLAGS_batch_size = batch_size_in_map->second;
  } else if (argc == 4) {
    FLAGS_batch_size = atoi(argv[3]);
  }
  xdims[0] = FLAGS_batch_size;
  rdims[0] = FLAGS_batch_size;

  // Load data into x and y
  float *x = allocate<float>(xdims);
  float *y = allocate<float>(rdims);
  loadData(x, y);

  // Load model
  float *conv1 = allocate<float>(conv1dims);
  float *conv2 = allocate<float>(conv2dims);
  float *fc1   = allocate<float>(fc1dims);
  float *fc2   = allocate<float>(fc2dims);
  loadModel(conv1, conv2, fc1, fc2);

  // Perform foward opertion
  int *out = zeros<int>(FLAGS_batch_size);

  // get start time
  const auto start = now();

  forward_operation(x, conv1, conv2, fc1, fc2, out);

  // get end time
  const auto end = now();

  // get elapsed time in milliseconds
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Get reference
  int *ref = zeros<int>(FLAGS_batch_size);
  argmax(y, rdims, ref);

  // Calculate correctness
  int num_correct = 0;
  for (const auto i : range(0, FLAGS_batch_size)) {
    if (out[i] == ref[i]) {
      num_correct++;
    }
  }
  std::cout << "Done with " << FLAGS_batch_size << " queries in "
            << "elapsed = " << elapsed << " milliseconds. Correctness: "
            << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

  delete[] x;
  delete[] y;
  delete[] conv1;
  delete[] conv2;
  delete[] fc1;
  delete[] fc2;
  delete[] out;
  delete[] ref;

  return 0;
}
