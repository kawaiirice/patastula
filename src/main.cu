#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>

#include <hdf5.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10
#define BLOCK_SIZE 16
#define TILE_WIDTH_1 16
#define TILE_WIDTH_2 8
#define BLOCK_SIZE_CONVO 64

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

// // From book chapter Figure 16.4
// static void conv_forward_valid(const float *X, const int xdims[4],
//                                const float *W, const int wdims[4], float *Y,
//                                const int ydims[4]) {
//   const auto filter_h   = wdims[0];
//   const auto filter_w   = wdims[1];
//   const auto in_channel = wdims[2];
//
//   for (const auto i : range(0, ydims[0])) {
//     for (const auto m : range(0, ydims[3])) {
//       for (const auto w : range(0, ydims[2])) {
//         for (const auto h : range(0, ydims[1])) {
//           for (const auto p : range(0, filter_h)) {
//             for (const auto q : range(0, filter_w)) {
//               for (const auto c : range(0, in_channel)) {
//                 const auto yoffset =
//                     ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
//                 const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
//                                      (h + p) * xdims[2] * xdims[3] +
//                                      (w + q) * xdims[3] + c;
//                 const auto woffset = p * wdims[1] * wdims[2] * wdims[3] +
//                                      q * wdims[2] * wdims[3] + c * wdims[3] + m;
//                 Y[yoffset] += X[xoffset] * W[woffset];
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }

// __global__ void conv_forward_valid_kernel(const float *X, const int x0, const int x1, const int x2, const int x3,
//                                const float *W, const int w0, const int w1, const int w2, const int w3, 
//                                float *Y, const int y0, const int y1, const int y2, const int y3) {
//   const auto filter_h   = w0;
//   const auto filter_w   = w1;
//   const auto in_channel = w2;

//   int i = blockIdx.x;
//   int m = blockIdx.y;
//   int idx = blockIdx.z*blockDim.x + threadIdx.x;
//   int h = idx/y2;
//   int w = idx%y2;
//   if(h < y1 && w < y2){
//     float acc = 0;
//     const auto yoffset = ((i * y1 + h) * y2 + w) * y3 + m;
//     for (const auto p : range(0, filter_h)) {
//       for (const auto q : range(0, filter_w)) {
//         for (const auto c : range(0, in_channel)) {
//           const auto xoffset = i * x1 * x2 * x3 + (h + p) * x2 * x3 + (w + q) * x3 + c;
//           const auto woffset = p * w1 * w2 * w3 + q * w2 * w3 + c * w3 + m;
//           // Y[yoffset] += X[xoffset] * W[woffset];
//           acc += X[xoffset] * W[woffset];
//         }
//       }
//     }
//     Y[yoffset] = acc;
//   }
// }

// __global__ void conv_forward_valid_shared_kernel_1(const float *X, const int x0, const int x1, const int x2, const int x3, 
//                                const float *W, const int w0, const int w1, const int w2, const int w3, 
//                                float *Y, const int y0, const int y1, const int y2, const int y3) {

//     const auto filter_h   = w0;
//     const auto in_channel = w2;

//     int W_grid = (y1-1)/TILE_WIDTH_1+1;

//     int X_TILE_WIDTH=TILE_WIDTH_1+filter_h-1;
//     extern __shared__ float  shmem[];
//     float *X_shared=&shmem[0];
//     float * W_shared=&shmem[X_TILE_WIDTH*X_TILE_WIDTH];

//     int n = blockIdx.x; // samples in batch
//     int m = blockIdx.y; // output feature map
//     int h_base = (blockIdx.z / W_grid) * TILE_WIDTH_1;
//     int w_base = (blockIdx.z % W_grid) * TILE_WIDTH_1;
//     int h_0 = threadIdx.y;
//     int w_0 = threadIdx.x;
//     int h = h_base + h_0; // output feature map height
//     int w = w_base + w_0; // output feature map width

    
//     if(h < y1 && w < y2){
//       float acc = 0;
//       for (const auto c : range(0, in_channel)){
//         // initialize w_shared
//         if(h_0 < 5 && w_0 < 5){
//           const auto woffset = h_0 * w1 * w2 * w3 +w_0 * w2 * w3 + c * w3 + m;
//           W_shared[h_0 * 5 + w_0]=W[woffset];
//         }
//         __syncthreads();

//         // initialize x_shared
        
//         for (int i = h; i < (h_base + X_TILE_WIDTH); i += TILE_WIDTH_1){
//           for(int j = w; j < (w_base + X_TILE_WIDTH); j += TILE_WIDTH_1){
//             // if(i<x1 && j<x2){
//             const auto xoffset = n * x1 * x2 * x3 +i * x2 * x3 +j * x3 + c;
//               X_shared[(i - h_base) * X_TILE_WIDTH + j - w_base] = X[xoffset];  
//             // } 
//             // else
//             //   X_shared[(i - h_base) * X_TILE_WIDTH + j - w_base]=0;
            
//             }   
//           }
        
//         __syncthreads();
        
//         // convolution
//         for(int p=0;p<5;p++){
//           for (int q=0;q<5;q++){
//             acc += X_shared[(h_0+p)*X_TILE_WIDTH+w_0+q]*W_shared[p*5+q];    
//           }
//         }
//         __syncthreads();
//       }
//       const auto yoffset = ((n * y1 + h) * y2 + w) * y3 + m;
//       Y[yoffset]=acc;       
//   }
// }

__global__ void conv_forward_valid_shared_kernel_2(const float *X, const int x0, const int x1, const int x2, const int x3, 
                               const float *W, const int w0, const int w1, const int w2, const int w3, 
                               float *Y, const int y0, const int y1, const int y2, const int y3) {

    const auto filter_h   = w0;
    const auto in_channel = w2;

    int W_grid = (y1-1)/TILE_WIDTH_2+1;

    int X_TILE_WIDTH=TILE_WIDTH_2+filter_h-1;
    extern __shared__ float  shmem[];
    float *X_shared=&shmem[0];
    float * W_shared=&shmem[X_TILE_WIDTH*X_TILE_WIDTH];

    int n = blockIdx.x; // samples in batch
    int m = blockIdx.y; // output feature map
    int h_base = (blockIdx.z / W_grid) * TILE_WIDTH_2;
    int w_base = (blockIdx.z % W_grid) * TILE_WIDTH_2;
    int h_0 = threadIdx.y;
    int w_0 = threadIdx.x;
    int h = h_base + h_0; // output feature map height
    int w = w_base + w_0; // output feature map width

    
    if(h < y1 && w < y2){
      float acc = 0;
      for (const auto c : range(0, in_channel)){
        // initialize w_shared
        if(h_0 < 5 && w_0 < 5){
          const auto woffset = h_0 * w1 * w2 * w3 +w_0 * w2 * w3 + c * w3 + m;
          W_shared[h_0 * 5 + w_0]=W[woffset];
        }
        __syncthreads();

        // initialize x_shared
        for (int i = h;i < h_base + X_TILE_WIDTH; i += TILE_WIDTH_2){
          for(int j = w;j < w_base + X_TILE_WIDTH; j += TILE_WIDTH_2){ 
            const auto xoffset = n * x1 * x2 * x3 +i * x2 * x3 +j * x3 + c;
              X_shared[(i-h_base)* X_TILE_WIDTH+j-w_base]=X[xoffset];
            }   
          }
        __syncthreads();
        
        // convolution
        for(int p=0;p<5;p++){
          for (int q=0;q<5;q++){
            acc += X_shared[(h_0+p)*X_TILE_WIDTH+w_0+q]*W_shared[p*5+q];    
          }
        }
        __syncthreads();
      }
      const auto yoffset = ((n * y1 + h) * y2 + w) * y3 + m;
      Y[yoffset]=acc;       
  }
}

// checked 
__global__ void conv_forward_valid_unrolled_X_kernel(const float *X, const int x0, const int x1, const int x2, const int x3, 
                                const float *W, const int w0, const int w1, const int w2, const int w3, 
                                float *Y, const int y0, const int y1) {
  
  int c, s, h_out, w_out, h_unroll, w_unroll, w_base;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int H_out = x1-w0+1;
  int W_out = x2-w0+1;
  int W_unroll = H_out * W_out;
  int C = x3;
  int K = w0;

  if (t < C * W_unroll) {
    c = t / W_unroll;
    s = t % W_unroll;
    h_out = s / W_out;
    w_out = s % W_out;
    h_unroll = h_out * W_out + w_out; 
    w_base = c * K * K;

    // unroll X
    for(int p = 0; p < K; p++) {
      for(int q = 0; q < K; q++) {
        w_unroll = w_base + p * K + q;
        const auto xoffset = /* n * x1 * x2 * x3 + */ (h_out+p) * x2 * x3 + (W_out+q) * x3 + c;
        Y[w_unroll*W_unroll+h_unroll] = X[xoffset]; 
      }
    }
  }
}

// checked
__global__ void conv_forward_valid_unrolled_W_kernel(const float *X, const int x0, const int x1, const int x2, const int x3, 
                                const float *W, const int w0, const int w1, const int w2, const int w3, 
                                float *Y, const int y0, const int y1) {

  int t = blockIdx.x * 64 + threadIdx.x;
  int m, c;
  // int h = t / w0;
  // int w = t % w0;
  // int c = w / w3;
  // int m = w % w3;
  int K = w0;
  if (t < w2*w3) {
    m = t / w2;
    c = t % w2;
    // c = t / w3;
    // m = t % w3;
    // unroll W
    for(int p = 0; p < K; p++) {
      for(int q = 0; q < K; q++) {
        // const auto woffset_before = (h+p) * w1 * w2 * w3 + (w+q) * x2 * x3 + c * x3 + m;
        // const auto woffset_after = m * y1 * y2 * y3 + c * y2 * y3 + (h+p) * y3 + (w+q);  
        const auto woffset_before = p * w1 * w2 * w3 + q * w2 * w3 + c * w3 + m;
        const auto woffset_after = m * w0 * w1 * w2 + c * w0 * w1 + p * w0 + q;
        Y[woffset_after] = W[woffset_before];
      }
    }
  }
}

__global__ void conv_forward_valid_unrolled_Y_kernel(const float *X, const int x0, const int x1, const int x2, const int x3, 
                                const float *W, const int w0, const int w1, const int w2, const int w3, 
                                float *Y, const int y0, const int y1, const int y2, const int y3){
  int t = blockIdx.x * 64 + threadIdx.x;
  int h, w;
  int H_out = x1-w0+1;
  int W_out = x2-w0+1;
  if(t < H_out*W_out){
    h = t / H_out;
    w = t % H_out;
    for(int p = 0; p < w3; p++){
      const auto offsetX = p * H_out * W_out + t;
      const auto offsetY = h * W_out * w3 + w * w3 + p;
      Y[offsetY] = X[offsetX];
    }
  } 
}


__global__ void relu_kernel(float *X, int size){
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  if(row < size)
    X[row] = (X[row] < 0) ? 0 : X[row];
}

// // Recified linear unit 4d
// static void relu4(float *X, const int xdims[4]) {
//   for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3])) {
//     X[i] = (X[i] < 0) ? 0 : X[i];
//   }
// }

// __global__ void relu4_kernel(float *X, const int xdims[4]){
//  int row = blockIdx.x*blockDim.x+threadIdx.x;
//  if(row < xdims[0] * xdims[1] * xdims[2] * xdims[3])
//    X[row] = (X[row] < 0) ? 0 : X[row];
// }

// // Recified linear unit 2d
// static void relu2(float *X, const int xdims[2]) {
//   for (const auto i : range(0, xdims[0] * xdims[1])) {
//     X[i] = (X[i] < 0) ? 0 : X[i];
//   }
// }

// __global__ void relu2_kernel(float *X, const int xdims[2]){
//    int i = blockIdx.x*blockDim.x+threadIdx.x;
//    if(i < xdims[0] * xdims[1])
//      X[i] = (X[i] < 0) ? 0 : X[i];
// }

// // From book chapter Figure 16.5
// static void average_pool(const float *X, const int xdims[4],
//                          const int pool_size, float *Y, const int ydims[4]) {
//   for (const auto i : range(0, ydims[0])) {
//     for (const auto m : range(0, ydims[3])) {
//       for (const auto w : range(0, ydims[2])) {
//         for (const auto h : range(0, ydims[1])) {
//           for (const auto p : range(0, pool_size)) {
//             for (const auto q : range(0, pool_size)) {
//               const auto yoffset =
//                   ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
//               const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
//                                    (pool_size * h + p) * xdims[2] * xdims[3] +
//                                    (pool_size * w + q) * xdims[3] + m;
//               Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
//             }
//           }
//         }
//       }
//     }
//   }
// }

__global__ void average_pool_kernel(const float *X, const int x0, const int x1, const int x2, const int x3, const int pool_size, 
                                    float *Y, const int y0, const int y1, const int y2, const int y3){
  int i = blockIdx.x;
  int m = blockIdx.y;
  int idx = blockIdx.z*blockDim.x + threadIdx.x;
  int h = idx/y2;
  int w = idx%y2;

  if(h < y1 && w < y2){
    const auto yoffset = ((i * y1 + h) * y2 + w) * y3 + m;
    float acc = 0;
    for (const auto p : range(0, pool_size)) {
      for (const auto q : range(0, pool_size)) {
        const auto xoffset = i * x1 * x2 * x3 + (pool_size * h + p) * x2 * x3 + (pool_size * w + q) * x3 + m;
        acc += X[xoffset] / (1.0f * pool_size * pool_size);
        // Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
      }
    }
    Y[yoffset] = acc;
  }
}

// static void fully_forward(const float *X, const int xdims[2], float *W,
//                           const int wdims[2], float *Y, const int ydims[2]) {
//   for (const auto i : range(0, xdims[0])) {
//     for (const auto j : range(0, wdims[1])) {
//       float sum = 0;
//       for (const auto k : range(0, xdims[1])) {
//         sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
//       }
//       Y[i * wdims[1] + j] = sum;
//     }
//   }
// }

__global__ void fully_forward_kernel (const float *X, const int x0, const int x1, 
                                      const float *W, const int w0, const int w1,
                                      float *Y, const int y0, const int y1) {
    int i, j;
    i = blockIdx.y * blockDim.y + threadIdx.y; // y index of output = row
    j = blockIdx.x * blockDim.x + threadIdx.x; // x index of output = col

    float sum = 0;
    __shared__ float Xs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Ws[BLOCK_SIZE][BLOCK_SIZE];

     for (const auto k : range(0, (x1-1)/BLOCK_SIZE + 1)) { //num X columns = num W rows
      if (k * BLOCK_SIZE + threadIdx.x < x1 && i < x0)
        Xs[threadIdx.y][threadIdx.x] = X[i * x1 + (k * BLOCK_SIZE + threadIdx.x)];
      else
        Xs[threadIdx.y][threadIdx.x] = 0.0;

      if (k * BLOCK_SIZE + threadIdx.y < w0 && j < w1)
        Ws[threadIdx.y][threadIdx.x] = W[(k * BLOCK_SIZE + threadIdx.y)*w1 + j];
      else
        Ws[threadIdx.y][threadIdx.x] = 0.0;

      __syncthreads();
      for (int p = 0; p < BLOCK_SIZE; ++p)
        sum += Xs[threadIdx.y][p] * Ws[p][threadIdx.x];
      __syncthreads();
    }

    if(i < y0 && j < y1)
      Y[i * y1 + j] = sum;

    // if(i < ydims[0] && j < ydims[1]){
    //   float sum = 0;
    //   for (const auto k : range(0, xdims[1])) {
    //     sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
    //   }
    //   Y[i * wdims[1] + j] = sum;
    // }
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

__global__ void argmax_kernel(const float *X, const int xdims[2], float *YVal, int *Y) {

  // /************ parallelizing outer loop ************/
  // int i= threadIdx.x;
  // auto max_idx = 0;
  // auto max     = input[i * fdims[1]];
  // for (const auto j : range(0, fdims[1])) {
  //   const auto elem = input[(i * fdims[1]) + j];
  //   if (elem > max) {
  //     max_idx = j;
  //     max     = elem;
  //   }
  // }
  // output[blockIdx.x*1024+i] = max_idx;

  /************ parallelizing inner loop ************/
  // __shared__ int max_idx[16][32];
  // __shared__ float max[16][32];
  __shared__ int max_idx[32];
  __shared__ float max[32];

  int y = blockIdx.y;
  int x = 2 * blockIdx.x * blockDim.x + threadIdx.x;

  int tx = threadIdx.x;
  // int ty = threadIdx.y;
  //
  // if(y < xdims[0] && x < xdims[1]) {
  //   max_idx[ty][tx] = x;
  //   max[ty][tx] = X[(y * xdims[1]) + x];
  // }
  // else {
  //   max_idx[ty][tx] = 0;
  //   max[ty][tx] = X[(y * xdims[1])]; //first of the row
  // }
  //
  // if(y < xdims[0] && x + blockDim.x < xdims[1]) {
  //   max_idx[ty][tx + blockDim.x] = x + blockDim.x;
  //   max[ty][tx + blockDim.x] = X[(y * xdims[1]) + x + blockDim.x];
  // }
  // else {
  //   max_idx[ty][tx + blockDim.x] = 0;
  //   max[ty][tx + blockDim.x] = X[(y * xdims[1])]; //first of the row
  // }
  if(x < xdims[1]) {
    max_idx[tx] = x;
    max[tx] = X[(y * xdims[1]) + x];
  }
  else {
    max_idx[tx] = 0;
    max[tx] = X[(y * xdims[1])]; //first of the row
  }

  if(x + blockDim.x < xdims[1]) {
    max_idx[tx + blockDim.x] = x + blockDim.x;
    max[tx + blockDim.x] = X[(y * xdims[1]) + x + blockDim.x];
  }
  else {
    max_idx[tx + blockDim.x] = 0;
    max[tx + blockDim.x] = X[(y * xdims[1])]; //first of the row
  }

  // for(int stride = blockDim.x; stride >= 1; stride >>= 1){
  //   __syncthreads();
  //   if(tx < stride){
  //     if(max[ty][tx] < max[ty][tx + stride]){
  //       max_idx[ty][tx] = max_idx[ty][tx + stride];
  //       max[ty][tx] = max[ty][tx + stride];
  //     }
  //   }
  // }
  for(int stride = blockDim.x; stride >= 1; stride >>= 1){
    __syncthreads();
    if(tx < stride){
      if(max[tx] < max[tx + stride]){
        max_idx[tx] = max_idx[tx + stride];
        max[tx] = max[tx + stride];
      }
    }
  }

  //__syncthreads();
  if(y < xdims[0] && YVal[y] < max[0]) {
      Y[y] = max_idx[0];
  }
}

// Forward operation for the CNN, a combination of conv layer + average pooling + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  /* Host */ 
  // data dimensions and size
  const int pool_size = 2;
  int H_out_1 = xdims[1]-conv1dims[0]+1;
  int W_out_1 = xdims[2]-conv1dims[0]+1;

  int device_x_size = xdims[0] * xdims[1] * xdims[2] * xdims[3];
  int device_conv1_size = conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3];
  int device_conv2_size = conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3];
  int device_fc1_size = fc1dims[0] * fc1dims[1];
  int device_fc2_size = fc2dims[0] * fc2dims[1];


  // to do
  const int adims_x[] = {conv1dims[0]*conv1dims[1]*conv1dims[2], H_out_1*W_out_1};
  int device_ax_size = H_out_1*W_out_1*conv1dims[0]*conv1dims[1]*conv1dims[2];
  
  const int adims_w[] = {conv1dims[3], conv1dims[0]*conv1dims[1]*conv1dims[2]};
  int device_aw_size = conv1dims[0]*conv1dims[1]*conv1dims[2]*conv1dims[3];

  const int adims_m[] = {conv1dims[3], H_out_1*W_out_1};
  // int device_am_size = xdims[1]*xdims[2]*conv1dims[2];
  int device_am_size = H_out_1*W_out_1*conv1dims[3]*xdims[0];

  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  int device_a_size = adims[0]*adims[1]*adims[2]*adims[3];

  const int bdims[] = {adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3]};
  int device_b_size = bdims[0]*bdims[1]*bdims[2]*bdims[3];

  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1), (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  int device_c_size = cdims[0]*cdims[1]*cdims[2]*cdims[3];

  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size, cdims[3]};
  int device_d_size = ddims[0]*ddims[1]*ddims[2]*ddims[3];

  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]}; // reshape for matrix mult section

  const int edims[] = {ddims[0], fc1dims[1]};
  int device_e_size = edims[0]*edims[1];

  const int fdims[] = {edims[0], fc2dims[1]};
  int device_f_size = fdims[0]*fdims[1];

  // allocating host memory
  auto a = zeros<float>(adims);
  auto b = zeros<float>(bdims);
  auto c = zeros<float>(cdims);
  auto d = zeros<float>(ddims);
  auto e = zeros<float>(edims);
  auto f = zeros<float>(fdims);

  /* Device */
  // device data
  float* device_x;
  float *device_a;
  float *device_ax;
  float *device_aw;
  float *device_am;
  float *device_b;
  float *device_c;
  float *device_d;
  float *device_e;
  float *device_f;
  // int *device_out;

  // device models
  float *device_conv1;
  float *device_conv2;
  float *device_fc1;
  float *device_fc2;

  // device malloc
  cudaMalloc((void **) &device_ax, device_ax_size * sizeof(float));
  cudaMalloc((void **) &device_aw, device_aw_size * sizeof(float));
  cudaMalloc((void **) &device_am, device_am_size * sizeof(float));
  cudaMalloc((void **) &device_x, device_x_size * sizeof(float));
  cudaMalloc((void **) &device_a, device_a_size * sizeof(float));
  cudaMalloc((void **) &device_b, device_b_size * sizeof(float));
  cudaMalloc((void **) &device_c, device_c_size * sizeof(float));
  cudaMalloc((void **) &device_d, device_d_size * sizeof(float));
  cudaMalloc((void **) &device_e, device_e_size * sizeof(float));
  cudaMalloc((void **) &device_f, device_f_size * sizeof(float));
  // cudaMalloc((void **) &device_out, FLAGS_batch_size * sizeof(int));

  cudaMalloc((void **) &device_conv1, device_conv1_size * sizeof(float));
  cudaMalloc((void **) &device_conv2, device_conv2_size * sizeof(float));
  cudaMalloc((void **) &device_fc1, device_fc1_size * sizeof(float));
  cudaMalloc((void **) &device_fc2, device_fc2_size * sizeof(float));

  // copy host to device
  cudaMemcpy(device_x, x, device_x_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_conv1, conv1, device_conv1_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_conv2, conv2, device_conv2_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_fc1, fc1, device_fc1_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_fc2, fc2, device_fc2_size * sizeof(float), cudaMemcpyHostToDevice);

  /* conv layer start */
  /************* unrolled convolution kernel begin *************/
  const auto start = now();
  // unroll X
  int num_threads = xdims[3]*H_out_1*W_out_1;
  int num_blocks = ceil((float)num_threads/512.0);
  dim3 DimGrid0X(num_blocks, 1, 1);
  dim3 DimBlock0X(512, 1, 1);
  conv_forward_valid_unrolled_X_kernel<<<DimGrid0X, DimBlock0X>>>(device_x, xdims[0], xdims[1], xdims[2], xdims[3], 
                                                    device_conv1, conv1dims[0], conv1dims[1], conv1dims[2], conv1dims[3], 
                                                    device_ax, adims_x[0], adims_x[1]);
  // float* ax_test;
  // ax_test = allocate<float>(adims_x);
  // cudaMemcpy(ax_test, device_ax, adims_x[0]*adims_x[1]*sizeof(float), cudaMemcpyDeviceToHost);
  // for(int test = 0; test < adims_x[0]; test++)
  //   for(int test1 = 0; test1 < adims_x[1]; test1++)
  //   std::cout << "device_ax: " << ax_test[test][test1] << "\n";

  // unroll W
  num_threads = conv1dims[2]*conv1dims[3];
  num_blocks = ceil((float)num_threads / 64.0);
  dim3 DimGrid0W(num_blocks, 1, 1);
  dim3 DimBlock0W(64, 1, 1);
  conv_forward_valid_unrolled_W_kernel<<<DimGrid0W, DimBlock0W>>>(device_x, xdims[0], xdims[1], xdims[2], xdims[3], 
                                                    device_conv1, conv1dims[0], conv1dims[1], conv1dims[2], conv1dims[3], 
                                                    device_aw, adims_w[0], adims_w[1]);
  cudaDeviceSynchronize();
  // float* aw_test;
  // aw_test = allocate<float>(adims_w);
  // cudaMemcpy(aw_test, device_aw, adims_w[0]*adims_w[1]*sizeof(float), cudaMemcpyDeviceToHost);
  // for(int test = 0; test < adims_w[0]*adims_w[1]; test++)
  //   std::cout << "device_aw: \n" << aw_test[test];

  // matrix multiplication
  dim3 DimGrid0M(ceil((float)(H_out_1 * W_out_1) / 16), ceil((float) conv1dims[3] / 16), 1);
  dim3 DimBlock0M(16, 16, 1);
  // for(int i = 0; i < xdims[0]; i++){
    // cudaMemcpy(device_x, &x[i*device_am_size], device_am_size, cudaMemcpyHostToDevice);
  fully_forward_kernel<<<DimGrid0M, DimBlock0M>>>(device_aw, adims_w[0], adims_w[1], 
                                                  device_ax, adims_x[0], adims_x[1], 
                                                  device_am, adims_m[0], adims_m[1]);
  // }
  // float* am_test;
  // am_test = allocate<float>(adims_m);
  // cudaMemcpy(am_test, device_am, adims_m[0]*adims_m[1]*sizeof(float), cudaMemcpyDeviceToHost);
  // for(int test = 0; test < adims_m[0]*adims_m[1]; test++)
  //   std::cout << "device_am: \n" << am_test[test];

  // unroll Y
  num_threads = H_out_1*W_out_1*conv1dims[3];
  dim3 DimGrid0Y(ceil((float)num_threads/1024.0), 1, 1);
  dim3 DimBlock0Y(1024, 1, 1);
  conv_forward_valid_unrolled_Y_kernel<<<DimGrid0Y, DimBlock0Y>>>(device_am, xdims[0], adims_x[0], adims[1], xdims[3], 
                                                                  device_conv1, conv1dims[0], conv1dims[1], conv1dims[2], conv1dims[3], 
                                                                  device_a, adims[0], adims[1], adims[2], adims[3]);

  /************* unrolled convolution kernel end *************/

  /************* tiled convolution kernel begin *************/
  int W_grid = (adims[1]-1)/TILE_WIDTH_1+1;
  int H_grid = (adims[2]-1)/TILE_WIDTH_1+1;

  // size_t shmem_size = sizeof(float) * ( (TILE_WIDTH_1 +conv1dims[0] -1)*(TILE_WIDTH_1 +conv1dims[1]-1) + conv1dims[0]*conv1dims[1] ); 
  // dim3 DimBlock(TILE_WIDTH_1, TILE_WIDTH_1, 1);
  // dim3 DimGrid(adims[0], adims[3], W_grid * H_grid);
  // const auto start = now();
  // conv_forward_valid_shared_kernel_1<<<DimGrid, DimBlock, shmem_size>>>(device_x, xdims[0], xdims[1], xdims[2], xdims[3], 
  //                                                   device_conv1, conv1dims[0], conv1dims[1], conv1dims[2], conv1dims[3], 
  //                                                   device_a, adims[0], adims[1], adims[2], adims[3]);
  /************* tiled convolution kernel end *************/
  
  /************* normal convolution kernel begin *************/
  // dim3 DimGrid(adims[0], adims[3], ceil(adims[1]*adims[2]/256.0));
  // dim3 DimBlock(256, 1, 1);
  // const auto start = now();
  // conv_forward_valid_kernel<<<DimGrid, DimBlock>>>(device_x, xdims[0], xdims[1], xdims[2], xdims[3], 
  //                                                   device_conv1, conv1dims[0], conv1dims[1], conv1dims[2], conv1dims[3], 
  //                                                   device_a, adims[0], adims[1], adims[2], adims[3]);
  /************* normal convolution kernel end *************/
  //conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);

  const auto end = now();
  const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Done with Conv1 in elapsed = " << elapsed << " milliseconds." << "\n";
  /* conv layer end*/

  /* relu kernel start */
  dim3 DimGrid1(ceil(device_a_size/256), 1, 1);
  dim3 DimBlock1(256, 1, 1);
  const auto start1 = now();
  relu_kernel<<<DimGrid1, DimBlock1>>>(device_a, device_a_size);
  // relu4(a, adims);
  const auto end1 = now();
  const auto elapsed1 = std::chrono::duration<double, std::milli>(end1 - start1).count();
  std::cout << "Done with relu4 elapsed = " << elapsed1 << " milliseconds." << "\n";
  /* relu kernel end */

  /* average pooling start */
  dim3 DimGrid2(bdims[0], bdims[3], ceil(bdims[1]*bdims[2]/256.0));
  dim3 DimBlock2(256, 1, 1);
  const auto start2 = now();
  average_pool_kernel<<<DimGrid2, DimBlock2>>>(device_a, adims[0], adims[1], adims[2], adims[3], pool_size, 
                                                device_b, bdims[0], bdims[1], bdims[2], bdims[3]);
  const auto end2 = now();
  const auto elapsed2 = std::chrono::duration<double, std::milli>(end2 - start2).count();
  std::cout << "Done with pool1 in elapsed = " << elapsed2 << " milliseconds." << "\n";
  /* average pooling end */

  /* conv layer start */
  // dim3 DimGrid3(cdims[0], cdims[3], ceil(cdims[1]*cdims[2]/256.0));
  // dim3 DimBlock3(256, 1, 1);
  W_grid = (cdims[1]-1)/TILE_WIDTH_2+1;
  H_grid = (cdims[2]-1)/TILE_WIDTH_2+1;

  size_t shmem_size3 = sizeof(float) * ( (TILE_WIDTH_2 +conv2dims[0] -1)*(TILE_WIDTH_2 +conv2dims[1]-1) + conv2dims[0]*conv2dims[1] ); 
  dim3 DimBlock3(TILE_WIDTH_2, TILE_WIDTH_2, 1);
  dim3 DimGrid3(cdims[0], cdims[3], W_grid * H_grid);
  const auto start3 = now();
  conv_forward_valid_shared_kernel_2<<<DimGrid3, DimBlock3, shmem_size3>>>(device_b, bdims[0], bdims[1], bdims[2], bdims[3], 
                                                    device_conv2, conv2dims[0], conv2dims[1], conv2dims[2], conv2dims[3], 
                                                    device_c, cdims[0], cdims[1], cdims[2], cdims[3]);
  // conv_forward_valid_kernel<<<DimGrid3, DimBlock3>>>(device_b, bdims[0], bdims[1], bdims[2], bdims[3], 
  //                                                   device_conv2, conv2dims[0], conv2dims[1], conv2dims[2], conv2dims[3], 
  //                                                   device_c, cdims[0], cdims[1], cdims[2], cdims[3]);
  const auto end3 = now();
  const auto elapsed3 = std::chrono::duration<double, std::milli>(end3 - start3).count();
  std::cout << "Done with Conv2 in elapsed = " << elapsed3 << " milliseconds." << "\n";
  /* conv layer end*/

  /* relu later start */
  dim3 DimGrid4(ceil(device_c_size/256), 1, 1);
  dim3 DimBlock4(256, 1, 1);
  const auto start4 = now();
  relu_kernel<<<DimGrid4, DimBlock4>>>(device_c, device_c_size);
  // relu4(c, cdims);
  const auto end4 = now();
  const auto elapsed4 = std::chrono::duration<double, std::milli>(end4 - start4).count();
  std::cout << "Done with relu4 in elapsed = " << elapsed4 << " milliseconds." << "\n";
  /* relu layer end*/

  /* average pooling start*/
  dim3 DimGrid5(ddims[0], ddims[3], ceil(ddims[1]*ddims[2]/256.0));
  dim3 DimBlock5(256, 1, 1);
  const auto start5 = now();
  average_pool_kernel<<<DimGrid5, DimBlock5>>>(device_c, cdims[0], cdims[1], cdims[2], cdims[3], pool_size, 
                                                device_d, ddims[0], ddims[1], ddims[2], ddims[3]);//, w_grid);
  //average_pool(c, cdims, pool_size, d, ddims);
  const auto end5 = now();
  const auto elapsed5 = std::chrono::duration<double, std::milli>(end5 - start5).count();
  std::cout << "Done with pool2 in elapsed = " << elapsed4 << " milliseconds." << "\n";
  /* average pooling end*/

  /* fully forward start */
  dim3 DimGridE((edims[1]-1)/BLOCK_SIZE + 1, (edims[0]-1)/BLOCK_SIZE + 1, 1);
  dim3 DimBlockE(BLOCK_SIZE, BLOCK_SIZE, 1);
  const auto start6 = now();
  fully_forward_kernel<<<DimGridE, DimBlockE>>>(device_d, ddims2[0], ddims2[1], device_fc1, fc1dims[0], fc1dims[1], device_e, edims[0], edims[1]); 
  // fully_forward(d, ddims2, fc1, fc1dims, e, edims);
  const auto end6 = now();
  const auto elapsed6 = std::chrono::duration<double, std::milli>(end6 - start6).count();
  std::cout << "Done with fully_forward1 in elapsed = " << elapsed6 << " milliseconds." << "\n";
  /* fully forward end */

  /* relu2 start */
  dim3 DimGridRelu2(ceil(device_e_size/256.0), 1, 1);
  dim3 DimBlockRelu2(256, 1, 1);
  const auto start7 = now();
  relu_kernel<<<DimGridRelu2, DimBlockRelu2>>>(device_e, device_e_size);
  // relu2(e, edims);
  const auto end7 = now();
  const auto elapsed7 = std::chrono::duration<double, std::milli>(end7 - start7).count();
  std::cout << "Done with relu2 in elapsed = " << elapsed7 << " milliseconds." << "\n";
  /* relu2 end */

  /* fully forward start */
  dim3 DimGridF((fdims[1] - 1)/BLOCK_SIZE + 1, (fdims[0] - 1)/BLOCK_SIZE + 1, 1);
  dim3 DimBlockF(BLOCK_SIZE, BLOCK_SIZE, 1);
  const auto start8 = now();
  fully_forward_kernel<<<DimGridF, DimBlockF>>>(device_e, edims[0], edims[1], device_fc2, fc2dims[0], fc2dims[1], device_f, fdims[0], fdims[1]);
  //fully_forward(e, edims, fc2, fc2dims, f, fdims);
  const auto end8 = now();
  const auto elapsed8 = std::chrono::duration<double, std::milli>(end8 - start8).count();
  std::cout << "Done with fully_forward 2 in elapsed = " << elapsed8 << " milliseconds." << "\n";
  /* fully forward end */

  // copy back to host 
  cudaMemcpy(f, device_f, device_f_size * sizeof(float), cudaMemcpyDeviceToHost);

  /* argmax start */
  const auto start9 = now();
  argmax(f, fdims, out);
  // argmax_kernel<<<DimGridArg, DimBlockArg>>>(device_f, device_fdims, deviceOutVal, deviceOut);
  const auto end9 = now();
  const auto elapsed9 = std::chrono::duration<double, std::milli>(end9 - start9).count();
  std::cout << "Done with argmax in " << "elapsed = " << elapsed << " milliseconds." << "\n";
  /* argmax end */

  // cudaMemcpy(out, deviceOut, fdims[0] * sizeof(int), cudaMemcpyDeviceToHost);

  // free device memory
  cudaFree(device_x);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  cudaFree(device_d);
  cudaFree(device_e);
  cudaFree(device_f);
  cudaFree(device_conv1);
  cudaFree(device_conv2);
  cudaFree(device_fc1);
  cudaFree(device_fc2);

  // free host memory
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  delete[] f;
}

int main(int argc, char **argv) {

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
  const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();

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
