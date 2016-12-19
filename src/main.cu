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
#define TILE_WIDTH 8

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

__global__ void conv_forward_valid_kernel(const float *X, const int x0, const int x1, const int x2, const int x3,
                               const float *W, const int w0, const int w1, const int w2, const int w3, 
                               float *Y, const int y0, const int y1, const int y2, const int y3) {
	const auto filter_h   = w0;
	const auto filter_w   = w1;
	const auto in_channel = w2;

	int i = blockIdx.x;
	int m = blockIdx.y;
	int idx = blockIdx.z*blockDim.x + threadIdx.x;
	int h = idx/y2;
	int w = idx%y2;
	if(h < y1 && w < y2){
    float acc = 0;
		const auto yoffset = ((i * y1 + h) * y2 + w) * y3 + m;
		for (const auto p : range(0, filter_h)) {
			for (const auto q : range(0, filter_w)) {
				for (const auto c : range(0, in_channel)) {
  				const auto xoffset = i * x1 * x2 * x3 + (h + p) * x2 * x3 + (w + q) * x3 + c;
  				const auto woffset = p * w1 * w2 * w3 + q * w2 * w3 + c * w3 + m;
  				// Y[yoffset] += X[xoffset] * W[woffset];
          acc += X[xoffset] * W[woffset];
				}
			}
		}
    Y[yoffset] = acc;
	}
}

__global__ void conv_forward_valid_shared_kernel(const float *X, const int x0, const int x1, const int x2, const int x3, 
                               const float *W, const int w0, const int w1, const int w2, const int w3, 
                               float *Y, const int y0, const int y1, const int y2, const int y3) {

    const auto filter_h   = w0;
    const auto in_channel = w2;

    int W_grid = (y1-1)/TILE_WIDTH+1;

    int X_TILE_WIDTH=TILE_WIDTH+filter_h-1;
    extern __shared__ float  shmem[];
    float *X_shared=&shmem[0];
    float * W_shared=&shmem[X_TILE_WIDTH*X_TILE_WIDTH];

    int n = blockIdx.x; // samples in batch
    int m = blockIdx.y; // output feature map
    int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
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
        for (int i = h;i < h_base + X_TILE_WIDTH; i += TILE_WIDTH){
          for(int j = w;j < w_base + X_TILE_WIDTH; j += TILE_WIDTH){ 
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
      Y[yoffset]= (acc < 0) ? 0: acc;       
  }
}

__global__ void conv_forward_valid_unrolled_kernel(const float *X, const int x0, const int x1, const int x2, const int x3, 
                               const float *W, const int w0, const int w1, const int w2, const int w3, 
                               float *Y, const int y0, const int y1, const int y2, const int y3) {
 
  extern __shared__ float shared_mem[];
  float *a_shared = &shared_mem[0];
  float *b_shared = &shared_mem[16 * 16];

  int n = blockIdx.z;
  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int tx = threadIdx.x; 
  int ty = threadIdx.y;

  float acc=0;
  for(int i=0; i<(x3*w0*w1+16-1)/16; ++i){
    if((by*16+ty<y3) && (i*16+tx < x3*w0*w1)){
      int p = ((i * 16 + tx) % (w0 * w1)) / w1;
      int q = ((i * 16 + tx) % (w0 * w1)) % w1;
      int c = (i * 16 + tx)/ (w0 * w1);
      int m = by * 16 + ty;
      a_shared[ty * 16 + tx]=W[p * w1 * w2 * w3 + q * w2 * w3 + c * w3 + m];
    }
    else
      a_shared[ty * 16 + tx]=0.0;

    if((i * 16 + ty<x3*w0*w1)&&(bx*16+tx<y1*y2)){
      int px = ((i * 16 + ty) % (w0 * w1)) / w1;
      int qx = ((i * 16 + ty) % (w0 * w1)) % w1;
      int cx = (i * 16 + ty) / (w0 * w1);
      int h = (bx * 16 + tx) / y2;
      int w = (bx * 16 + tx) % y2;
      b_shared[ty * 16 + tx] = X[n * x1 * x2 * x3 + (h + px) * x2 * x3 +(w + qx) * x3 + cx];
    }
    else
      b_shared[ty * 16 + tx]=0.0;

    __syncthreads();
    
    for(int k=0;k<16;++k){
      acc+=a_shared[ty * 16 + k]*b_shared[k * 16 + tx];
    }
    __syncthreads();
  }

  if ((by*16+ty<y3)&&(bx*16+tx<y1*y2)&&(acc>0)){
    atomicAdd(&Y[n * y1 * y2 * y3 + ((bx*16+tx )/ y2) * y2 * y3 + ((bx*16+tx) % y2) * y3 + by*16+ty], acc);
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
// 	int row = blockIdx.x*blockDim.x+threadIdx.x;
// 	if(row < xdims[0] * xdims[1] * xdims[2] * xdims[3])
// 		X[row] = (X[row] < 0) ? 0 : X[row];
// }

// // Recified linear unit 2d
// static void relu2(float *X, const int xdims[2]) {
//   for (const auto i : range(0, xdims[0] * xdims[1])) {
//     X[i] = (X[i] < 0) ? 0 : X[i];
//   }
// }

// __global__ void relu2_kernel(float *X, const int xdims[2]){
//  	int i = blockIdx.x*blockDim.x+threadIdx.x;
//  	if(i < xdims[0] * xdims[1])
//  		X[i] = (X[i] < 0) ? 0 : X[i];
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
	int m = threadIdx.x + blockDim.x*blockIdx.z;
	int idx = blockIdx.y;
	int h = idx/y2;
	int w = idx%y2;

	if(h < y1 && w < y2 && m < y3){
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

__global__ void fully_forward_kernel(const float *X, const int x0, const int x1, 
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

    // basic matrix mult
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

__global__ void argmax_kernel1(const float *X, const int x0, const int x1, int *Y) {
  int i= blockIdx.x * blockDim.x + threadIdx.x;
  
  if(i < x0/2) {
    auto max_idx = 0;
    auto max = X[i * x1];
    for (const auto j : range(0, x1)) {
      const auto elem = X[(i * x1) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
}
__global__ void argmax_kernel2(const float *X, const int x0, const int x1, int *Y) {
  int i= blockIdx.x * blockDim.x + threadIdx.x;
  
  if(i < x0 && i >= x0/2) {
    auto max_idx = 0;
    auto max = X[i * x1];
    for (const auto j : range(0, x1)) {
      const auto elem = X[(i * x1) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
}

// Forward operation for the CNN, a combination of conv layer + average pooling + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out,
					   float *device_x, float *device_a, float *device_b, float *device_c,
					   float *device_d, float *device_e, float *device_f, int *device_out,
					   float *device_conv1, float *device_conv2, float *device_fc1, float *device_fc2){
  /* Host */ 
  // data dimensions and size
  const int pool_size = 2;

  int device_x_size = xdims[0] * xdims[1] * xdims[2] * xdims[3];
  int device_conv1_size = conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3];
  int device_conv2_size = conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3];
  int device_fc1_size = fc1dims[0] * fc1dims[1];
  int device_fc2_size = fc2dims[0] * fc2dims[1];

  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  int device_a_size = adims[0]*adims[1]*adims[2]*adims[3];

  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3]};

  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1), (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  int device_c_size = cdims[0]*cdims[1]*cdims[2]*cdims[3];

  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size, cdims[3]};

  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]}; // reshape for matrix mult section

  const int edims[] = {ddims[0], fc1dims[1]};
  int device_e_size = edims[0]*edims[1];

  const int fdims[] = {edims[0], fc2dims[1]};

  // allocating host memory
  // auto a = zeros<float>(adims);
  // auto b = zeros<float>(bdims);
  // auto c = zeros<float>(cdims);
  // auto d = zeros<float>(ddims);
  // auto e = zeros<float>(edims);
  // auto f = zeros<float>(fdims);

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  /* Device */
  // device data

  // pin memory
  // cudaHostRegister(out, FLAGS_batch_size * sizeof(int), cudaHostRegisterDefault);
  cudaHostRegister(x, device_x_size * sizeof(float), cudaHostRegisterDefault);
  cudaHostRegister(conv1, device_conv1_size * sizeof(float), cudaHostRegisterDefault);
  cudaHostRegister(conv2, device_conv2_size * sizeof(float), cudaHostRegisterDefault);
  cudaHostRegister(fc1, device_fc1_size * sizeof(float), cudaHostRegisterDefault);
  cudaHostRegister(fc2, device_fc2_size * sizeof(float), cudaHostRegisterDefault);

  // copy host to device
  cudaMemcpy(device_x, x, device_x_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_conv1, conv1, device_conv1_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_conv2, conv2, device_conv2_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_fc1, fc1, device_fc1_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_fc2, fc2, device_fc2_size * sizeof(float), cudaMemcpyHostToDevice);

  //unpin memory
  cudaHostUnregister(x);
  cudaHostUnregister(conv1);
  cudaHostUnregister(conv2);
  cudaHostUnregister(fc1);
  cudaHostUnregister(fc2);

  
  /* conv layer start */
   int W_grid, H_grid;
   W_grid = (adims[1]-1)/TILE_WIDTH+1;
   H_grid = (adims[2]-1)/TILE_WIDTH+1;

   size_t shmem_size = sizeof(float) * ( (TILE_WIDTH +conv1dims[0] -1)*(TILE_WIDTH +conv1dims[1]-1) + conv1dims[0]*conv1dims[1] ); 
   dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
   dim3 DimGrid(adims[0], adims[3], W_grid * H_grid);
  const auto start = now();
   conv_forward_valid_shared_kernel<<<DimGrid, DimBlock, shmem_size>>>(device_x, xdims[0], xdims[1], xdims[2], xdims[3], 
                                                     device_conv1, conv1dims[0], conv1dims[1], conv1dims[2], conv1dims[3], 
                                                     device_a, adims[0], adims[1], adims[2], adims[3]);
   /*
   dim3 DimGrid(adims[0], adims[3], ceil(adims[1]*adims[2]/256.0));
   dim3 DimBlock(256, 1, 1);
   const auto start = now();
   conv_forward_valid_kernel<<<DimGrid, DimBlock>>>(device_x, xdims[0], xdims[1], xdims[2], xdims[3], 
                                                     device_conv1, conv1dims[0], conv1dims[1], conv1dims[2], conv1dims[3], 
                                                     device_a, adims[0], adims[1], adims[2], adims[3]);
   */

  /*
  auto a_zero = zeros<float>(adims);
  cudaMemcpy(device_a, a_zero, device_a_size * sizeof(float), cudaMemcpyHostToDevice);
  dim3 DimBlock(16, 16, 1);
  dim3 DimGrid((adims[1] * adims[2] + 16-1)/16, (adims[3]+16-1)/16, adims[0]);
  size_t shmem_size = sizeof(float) * (16 * 16 * 2);
  conv_forward_valid_unrolled_kernel<<<DimGrid, DimBlock, shmem_size>>>(device_x, xdims[0], xdims[1], xdims[2], xdims[3], 
                                                    device_conv1, conv1dims[0], conv1dims[1], conv1dims[2], conv1dims[3], 
                                                    device_a, adims[0], adims[1], adims[2], adims[3]);
*/

  //conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);
  const auto end = now();
  const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Done with Conv1 in elapsed = " << elapsed << " milliseconds." << "\n";
  /* conv layer end*/

  // /* relu kernel start */
   //dim3 DimGrid1(ceil(device_a_size/256), 1, 1);
   //dim3 DimBlock1(256, 1, 1);
  // const auto start1 = now();
   //relu_kernel<<<DimGrid1, DimBlock1>>>(device_a, device_a_size);
  // // relu4(a, adims);
  // const auto end1 = now();
  // const auto elapsed1 = std::chrono::duration<double, std::milli>(end1 - start1).count();
  // std::cout << "Done with relu4 elapsed = " << elapsed1 << " milliseconds." << "\n";
  // /* relu kernel end */

  /* average pooling start */
  dim3 DimGrid2(bdims[0],bdims[1]*bdims[2], ceil(bdims[3]/32.0));
  dim3 DimBlock2(32, 1, 1);
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
  // W_grid = (cdims[1]-1)/TILE_WIDTH+1;
  // H_grid = (cdims[2]-1)/TILE_WIDTH+1;

  // size_t shmem_size3 = sizeof(float) * ( (TILE_WIDTH +conv2dims[0] -1)*(TILE_WIDTH +conv2dims[1]-1) + conv2dims[0]*conv2dims[1] ); 
  // dim3 DimBlock3(TILE_WIDTH, TILE_WIDTH, 1);
  // dim3 DimGrid3(cdims[0], cdims[3], W_grid * H_grid);
  const auto start3 = now();
  //   conv_forward_valid_shared_kernel<<<DimGrid3, DimBlock3, shmem_size3>>>(device_b, bdims[0], bdims[1], bdims[2], bdims[3], 
  //                                                   device_conv2, conv2dims[0], conv2dims[1], conv2dims[2], conv2dims[3], 
  //                                                   device_c, cdims[0], cdims[1], cdims[2], cdims[3]);
  // conv_forward_valid_kernel<<<DimGrid3, DimBlock3>>>(device_b, bdims[0], bdims[1], bdims[2], bdims[3], 
  //                                                   device_conv2, conv2dims[0], conv2dims[1], conv2dims[2], conv2dims[3], 
  //                                                   device_c, cdims[0], cdims[1], cdims[2], cdims[3]);
  auto c_zero = zeros<float>(cdims);
  cudaMemcpy(device_c, c_zero, device_c_size * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemset(device_c, 0, device_c_size);
  dim3 DimBlock3(16, 16, 1);
  dim3 DimGrid3((cdims[1] * cdims[2] + 16-1)/16, (cdims[3]+16-1)/16, cdims[0]);
  shmem_size = sizeof(float) * (16 * 16 * 2);
  conv_forward_valid_unrolled_kernel<<<DimGrid3, DimBlock3, shmem_size>>>(device_b, bdims[0], bdims[1], bdims[2], bdims[3], 
                                                    device_conv2, conv2dims[0], conv2dims[1], conv2dims[2], conv2dims[3], 
                                                    device_c, cdims[0], cdims[1], cdims[2], cdims[3]);
  const auto end3 = now();
  const auto elapsed3 = std::chrono::duration<double, std::milli>(end3 - start3).count();
  std::cout << "Done with Conv2 in elapsed = " << elapsed3 << " milliseconds." << "\n";
  /* conv layer end*/

  // /* relu later start */
   //dim3 DimGrid4(ceil(device_c_size/256), 1, 1);
   //dim3 DimBlock4(256, 1, 1);
  // const auto start4 = now();
   //relu_kernel<<<DimGrid4, DimBlock4>>>(device_c, device_c_size);
  // // relu4(c, cdims);
  // const auto end4 = now();
  // const auto elapsed4 = std::chrono::duration<double, std::milli>(end4 - start4).count();
  // std::cout << "Done with relu4 in elapsed = " << elapsed4 << " milliseconds." << "\n";
  // /* relu layer end*/

  /* average pooling start*/
  dim3 DimGrid5(ddims[0], ddims[1]*ddims[2], ceil(ddims[3]/32.0));
  dim3 DimBlock5(32, 1, 1);
  const auto start5 = now();
  average_pool_kernel<<<DimGrid5, DimBlock5>>>(device_c, cdims[0], cdims[1], cdims[2], cdims[3], pool_size, 
                                                device_d, ddims[0], ddims[1], ddims[2], ddims[3]);//, w_grid);
  //average_pool(c, cdims, pool_size, d, ddims);
  const auto end5 = now();
  const auto elapsed5 = std::chrono::duration<double, std::milli>(end5 - start5).count();
  std::cout << "Done with pool2 in elapsed = " << elapsed5 << " milliseconds." << "\n";
  /* average pooling end*/

  /* fully forward start */
  dim3 DimGrid6((edims[1]-1)/BLOCK_SIZE + 1, (edims[0]-1)/BLOCK_SIZE + 1, 1);
  dim3 DimBlock6(BLOCK_SIZE, BLOCK_SIZE, 1);
  const auto start6 = now();
  fully_forward_kernel<<<DimGrid6, DimBlock6>>>(device_d, ddims2[0], ddims2[1], device_fc1, fc1dims[0], fc1dims[1], device_e, edims[0], edims[1]); 
  // fully_forward(d, ddims2, fc1, fc1dims, e, edims);
  const auto end6 = now();
  const auto elapsed6 = std::chrono::duration<double, std::milli>(end6 - start6).count();
  std::cout << "Done with fully_forward1 in elapsed = " << elapsed6 << " milliseconds." << "\n";
  /* fully forward end */

  /* relu2 start */
  dim3 DimGrid7(ceil(device_e_size/256.0), 1, 1);
  dim3 DimBlock7(256, 1, 1);
  const auto start7 = now();
  relu_kernel<<<DimGrid7, DimBlock7>>>(device_e, device_e_size);
  // relu2(e, edims);
  const auto end7 = now();
  const auto elapsed7 = std::chrono::duration<double, std::milli>(end7 - start7).count();
  std::cout << "Done with relu2 in elapsed = " << elapsed7 << " milliseconds." << "\n";
  /* relu2 end */

  /* fully forward start */
  dim3 DimGrid8((fdims[1] - 1)/BLOCK_SIZE + 1, (fdims[0] - 1)/BLOCK_SIZE + 1, 1);
  dim3 DimBlock8(BLOCK_SIZE, BLOCK_SIZE, 1);
  const auto start8 = now();
  fully_forward_kernel<<<DimGrid8, DimBlock8>>>(device_e, edims[0], edims[1], device_fc2, fc2dims[0], fc2dims[1], device_f, fdims[0], fdims[1]);
  //fully_forward(e, edims, fc2, fc2dims, f, fdims);
  const auto end8 = now();
  const auto elapsed8 = std::chrono::duration<double, std::milli>(end8 - start8).count();
  std::cout << "Done with fully_forward 2 in elapsed = " << elapsed8 << " milliseconds." << "\n";
  /* fully forward end */

  // copy back to host 
  // const auto start0 = now();
  // // cudaMemcpy(f, device_f, device_f_size * sizeof(float), cudaMemcpyDeviceToHost);
  // const auto end0 = now();
  // const auto elapsed0 = std::chrono::duration<double, std::milli>(end0 - start0).count();
  // std::cout << "Copy back to host in elapsed = " << elapsed0 << " milliseconds." << "\n";

  /* argmax start */
  dim3 DimGrid9((FLAGS_batch_size - 1)/16 + 1, 1, 1);
  dim3 DimBlock9(16, 1, 1);
  const auto start9 = now();
  // argmax(f, fdims, out);
  argmax_kernel1<<<DimGrid9, DimBlock9, 0, stream1>>>(device_f, fdims[0], fdims[1], device_out);
  argmax_kernel2<<<DimGrid9, DimBlock9, 0, stream2>>>(device_f, fdims[0], fdims[1], device_out);
  const auto end9 = now();
  const auto elapsed9 = std::chrono::duration<double, std::milli>(end9 - start9).count();
  std::cout << "Done with argmax in " << "elapsed = " << elapsed << " milliseconds." << "\n";
  /* argmax end */
  // cudaHostRegister(out, FLAGS_batch_size * sizeof(int), cudaHostRegisterDefault);
  // copy back to host 
  const auto start0 = now();
  cudaMemcpyAsync(out, device_out, FLAGS_batch_size/2 * sizeof(int), cudaMemcpyDeviceToHost, stream1);
  cudaMemcpyAsync(&out[FLAGS_batch_size/2], &device_out[FLAGS_batch_size/2], FLAGS_batch_size/2 * sizeof(int), cudaMemcpyDeviceToHost, stream2);
  const auto end0 = now();
  const auto elapsed0 = std::chrono::duration<double, std::milli>(end0 - start0).count();
  std::cout << "Copy back to host in elapsed = " << elapsed0 << " milliseconds." << "\n";

  // cudaHostUnregister(out);


  // free host memory
  // delete[] a;
  // delete[] b;
  // delete[] c;
  // delete[] d;
  // delete[] e;
  // delete[] f;
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
  
  // Malloc for forward_operation
  // data dimensions and size
  const int pool_size = 2;

  int device_x_size = xdims[0] * xdims[1] * xdims[2] * xdims[3];
  int device_conv1_size = conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3];
  int device_conv2_size = conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3];
  int device_fc1_size = fc1dims[0] * fc1dims[1];
  int device_fc2_size = fc2dims[0] * fc2dims[1];

  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  int device_a_size = adims[0]*adims[1]*adims[2]*adims[3];

  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3]};
  int device_b_size = bdims[0]*bdims[1]*bdims[2]*bdims[3];

  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1), (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  int device_c_size = cdims[0]*cdims[1]*cdims[2]*cdims[3];

  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size, cdims[3]};
  int device_d_size = ddims[0]*ddims[1]*ddims[2]*ddims[3];


  const int edims[] = {ddims[0], fc1dims[1]};
  int device_e_size = edims[0]*edims[1];

  const int fdims[] = {edims[0], fc2dims[1]};
  int device_f_size = fdims[0]*fdims[1];
  /* Device */
  // device data
  float* device_x;
  float *device_a;
  float *device_b;
  float *device_c;
  float *device_d;
  float *device_e;
  float *device_f;
  int *device_out;

  // device models
  float *device_conv1;
  float *device_conv2;
  float *device_fc1;
  float *device_fc2;

  // device malloc
  cudaMalloc((void **) &device_x, device_x_size * sizeof(float));
  cudaMalloc((void **) &device_a, device_a_size * sizeof(float));
  cudaMalloc((void **) &device_b, device_b_size * sizeof(float));
  cudaMalloc((void **) &device_c, device_c_size * sizeof(float));
  cudaMalloc((void **) &device_d, device_d_size * sizeof(float));
  cudaMalloc((void **) &device_e, device_e_size * sizeof(float));
  cudaMalloc((void **) &device_f, device_f_size * sizeof(float));
  cudaMalloc((void **) &device_out, FLAGS_batch_size * sizeof(int));

  cudaMemset(device_a, 0, device_a_size);
  cudaMemset(device_c, 0, device_c_size);

  cudaMalloc((void **) &device_conv1, device_conv1_size * sizeof(float));
  cudaMalloc((void **) &device_conv2, device_conv2_size * sizeof(float));
  cudaMalloc((void **) &device_fc1, device_fc1_size * sizeof(float));
  cudaMalloc((void **) &device_fc2, device_fc2_size * sizeof(float));

  // get start time
  const auto start = now();

  forward_operation(x, conv1, conv2, fc1, fc2, out,
					device_x, device_a, device_b, device_c, device_d, device_e, device_f, device_out,
					device_conv1, device_conv2, device_fc1, device_fc2);

  // get end time
  const auto end = now();

  // free device memory
  cudaFree(device_x);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  cudaFree(device_d);
  cudaFree(device_e);
  cudaFree(device_f);
  cudaFree(device_out);
  cudaFree(device_conv1);
  cudaFree(device_conv2);
  cudaFree(device_fc1);
  cudaFree(device_fc2);

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
