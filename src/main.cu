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
#define BLOCK_SIZE 32

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

__global__ void conv_forward_valid_kernel(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
	const auto filter_h   = wdims[0];
	const auto filter_w   = wdims[1];
	const auto in_channel = wdims[2];

	int i = blockIdx.x;
	int m = blockIdx.y;
	int idx = blockIdx.z*blockDim.x + threadIdx.x;
	int h = idx/ydims[2];
	int w = idx%ydims[2];
	if(h < ydims[1] && w < ydims[2]){
    // float acc = 0;
		const auto yoffset = ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
		for (const auto p : range(0, filter_h)) {
			for (const auto q : range(0, filter_w)) {
				for (const auto c : range(0, in_channel)) {
  				const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
  								 (h + p) * xdims[2] * xdims[3] +
  								 (w + q) * xdims[3] + c;
  				const auto woffset = p * wdims[1] * wdims[2] * wdims[3] +
  								 q * wdims[2] * wdims[3] + c * wdims[3] + m;
  				Y[yoffset] += X[xoffset] * W[woffset];
        // acc += X[xoffset] * W[woffset];
				}
			}
		}
    // Y[yoffset] = acc;
	}
}

// // Recified linear unit 4d
// static void relu4(float *X, const int xdims[4]) {
//   for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3])) {
//     X[i] = (X[i] < 0) ? 0 : X[i];
//   }
// }

__global__ void relu4_kernel(float *X, const int xdims[4]){
	int row = blockIdx.x*blockDim.x+threadIdx.x;
	if(row < xdims[0] * xdims[1] * xdims[2] * xdims[3])
		X[row] = (X[row] < 0) ? 0 : X[row];
}

// // Recified linear unit 2d
// static void relu2(float *X, const int xdims[2]) {
//   for (const auto i : range(0, xdims[0] * xdims[1])) {
//     X[i] = (X[i] < 0) ? 0 : X[i];
//   }
// }

__global__ void relu2_kernel(float *X, const int xdims[2]){
 	int i = blockIdx.x*blockDim.x+threadIdx.x;
 	if(i < xdims[0] * xdims[1])
 		X[i] = (X[i] < 0) ? 0 : X[i];
}

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

__global__ void average_pool_kernel(const float *X, const int xdims[4],
                         const int pool_size, float *Y, const int ydims[4]){//, const int w_grid) {
	int i = blockIdx.x;
	int m = blockIdx.y;
	int idx = blockIdx.z*blockDim.x + threadIdx.x;
	int h = idx/ydims[2];
	int w = idx%ydims[2];
	//int h = blockIdx.z/(w_grid) + threadIdx.y;
	//int w = blockIdx.z%(w_grid) + threadIdx.x;
	if(h < ydims[1] && w < ydims[2]){
	  const auto yoffset = ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
	  //float acc = 0;
	  for (const auto p : range(0, pool_size)) {
		for (const auto q : range(0, pool_size)) {
		  const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
							   (pool_size * h + p) * xdims[2] * xdims[3] +
							   (pool_size * w + q) * xdims[3] + m;
		  //acc += X[xoffset] / (1.0f * pool_size * pool_size);
		  Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
		  //atomicAdd(&Y[yoffset], X[xoffset]/(1.0f * pool_size * pool_size));
		}
	  }
	  //Y[yoffset] = acc;
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

__global__ void fully_forward_kernel (const float *X, const int xdims[2], float *W,
                          const int wdims[2], float *Y, const int ydims[2]) {
    int i, j;
    i = blockIdx.y * blockDim.y + threadIdx.y; // y index of output = row
    j = blockIdx.x * blockDim.x + threadIdx.x; // x index of output = col

    float sum = 0;
    __shared__ float Xs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Ws[BLOCK_SIZE][BLOCK_SIZE];

     for (const auto k : range(0, (xdims[1]-1)/BLOCK_SIZE + 1)) { //num X columns = num W rows
      if (k * BLOCK_SIZE + threadIdx.x < xdims[1] && i < xdims[0])
        Xs[threadIdx.y][threadIdx.x] = X[i * xdims[1] + (k * BLOCK_SIZE + threadIdx.x)];
      else
        Xs[threadIdx.y][threadIdx.x] = 0.0;

      if (k * BLOCK_SIZE + threadIdx.y < wdims[0] && j < wdims[1])
        Ws[threadIdx.y][threadIdx.x] = W[(k * BLOCK_SIZE + threadIdx.y)*wdims[1] + j];
      else
        Ws[threadIdx.y][threadIdx.x] = 0.0;

      __syncthreads();
      for (int p = 0; p < BLOCK_SIZE; ++p)
        sum += Xs[threadIdx.y][p] * Ws[p][threadIdx.x];
      __syncthreads();
    }

    if(i < ydims[0] && j < ydims[1])
      Y[i * ydims[1] + j] = sum;

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

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  // variables
  const int pool_size = 2;
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  int device_a_size = adims[0]*adims[1]*adims[2]*adims[3];

  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3]};
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
  // auto b = zeros<float>(bdims);
  // auto c = zeros<float>(cdims);
  // auto d = zeros<float>(ddims);
  // auto e = zeros<float>(edims);
  // auto f = zeros<float>(fdims);

  float *device_x;
  int device_x_size = xdims[0] * xdims[1] * xdims[2] * xdims[3];

  cudaMalloc((void **)&device_x, sizeof(float)*device_b_size);
  cudaMemcpy(device_x, x, sizeof(float)*device_x_size, cudaMemcpyHostToDevice);

  int *device_xdims;
  cudaMalloc((void **)&device_xdims, sizeof(int)*4);
  cudaMemcpy(device_xdims, xdims, sizeof(int)*4, cudaMemcpyHostToDevice);

  float *device_conv1;
  int device_conv1_size = conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3];
  int device_conv2_size = conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3];
  cudaMalloc((void **)&device_conv1, sizeof(float)*fc1dims[0] * fc1dims[1]);
  cudaMemcpy(device_conv1, conv1, sizeof(float)*device_conv1_size, cudaMemcpyHostToDevice);

  int *device_conv1dims;
  cudaMalloc((void **)&device_conv1dims, sizeof(int)*4);
  cudaMemcpy(device_conv1dims, conv1dims, sizeof(int)*4, cudaMemcpyHostToDevice);

  float *device_a;
  int *device_adims;

  cudaMalloc((void **)&device_adims, sizeof(int)*4);
  cudaMemcpy(device_adims, adims, sizeof(int)*4, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_a, sizeof(float)* device_a_size);
  cudaMemcpy(device_a, a, sizeof(float)*device_a_size, cudaMemcpyHostToDevice);


  dim3 DimGrid0(adims[0], adims[3], ceil(adims[1]*adims[2]/256.0));
  dim3 DimBlock0(256, 1, 1);

  // conv layer
  const auto start = now();
  conv_forward_valid_kernel<<<DimGrid0, DimBlock0>>>(device_x, device_xdims, device_conv1, device_conv1dims, device_a, device_adims);
  //conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);
  const auto end = now();
  const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Done with " << adims[0] << " queries in Conv1 "
          << "elapsed = " << elapsed << " milliseconds." << "\n";
  // cudaMemcpy(a, device_a, sizeof(float)*device_a_size, cudaMemcpyDeviceToHost);


  /// relu layer
  //relu4(a, adims);

  // relu kernel start layer
  /*
  float *device_a;
  int *device_adims;
  int device_a_size = adims[0]*adims[1]*adims[2]*adims[3];

  cudaMalloc((void **)&device_adims, sizeof(int)*4);
  cudaMemcpy(device_adims, adims, sizeof(int)*4, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_a, sizeof(float)* device_a_size);
  cudaMemcpy(device_a, a, sizeof(float)*device_a_size, cudaMemcpyHostToDevice);
  */

  dim3 DimGrid(ceil(device_a_size/256), 1, 1);
  dim3 DimBlock(256, 1, 1);
  const auto start1 = now();
  relu4_kernel<<<DimGrid, DimBlock>>>(device_a, device_adims);
  // relu4(a, adims);
  const auto end1 = now();
  const auto elapsed1 = std::chrono::duration<double, std::milli>(end1 - start1).count();
  std::cout << "Done with relu4 elapsed = " << elapsed1 << " milliseconds." << "\n";
  // cudaMemcpy(device_a, a, sizeof(float)*device_a_size, cudaMemcpyHostToDevice);

  //cudaMemcpy(device_adims, adims, sizeof(int)*4, cudaMemcpyDeviceToHost);
  //cudaMemcpy(a, device_a, sizeof(float)*device_a_size, cudaMemcpyDeviceToHost);
  // relu kernel end layer

  // average pooling
  //const int pool_size = 2;
  //const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3]};
  auto b = zeros<float>(bdims);
  //average_pool(a, adims, pool_size, b, bdims);

  //std::cout << "Device a and c sizes: " << device_a_size << " " << device_c_size << std::endl;

  cudaMemcpy(device_xdims, bdims, sizeof(int)*4, cudaMemcpyHostToDevice);
  cudaMemcpy(device_x, b, sizeof(float)*device_b_size, cudaMemcpyHostToDevice);

  // int tile_size = 16;
  // int w_grid = ceil(bdims[2]/tile_size);
  // int h_grid = ceil(bdims[1]/tile_size);

  dim3 DimGrid1(bdims[0], bdims[3], ceil(bdims[1]*bdims[2]/256.0));
  dim3 DimBlock1(256, 1, 1);

  const auto start2 = now();
  average_pool_kernel<<<DimGrid1, DimBlock1>>>(device_a, device_adims, pool_size, device_x, device_xdims);//, w_grid);
  const auto end2 = now();
  const auto elapsed2 = std::chrono::duration<double, std::milli>(end2 - start2).count();
  std::cout << "Done with " << adims[0] << " queries in pool1 "
          << "elapsed = " << elapsed2 << " milliseconds." << "\n";
  //cudaMemcpy(b, device_x, sizeof(float)*device_b_size, cudaMemcpyDeviceToHost);

  /*
  // average pooling kernel start
  float *device_b;
  int *device_bdims;
  int device_b_size = bdims[0]*bdims[1]*bdims[2]*bdims[3];

  cudaMalloc((void **)&device_bdims, sizeof(int)*4);
  cudaMemcpy(device_bdims, bdims, sizeof(int)*4, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_b, sizeof(float)* device_b_size);
  cudaMemcpy(device_b, b, sizeof(float)*device_b_size, cudaMemcpyHostToDevice);

  int tile_size = 16;
  int w_grid = ceil(bdims[2]/tile_size);
  int h_grid = ceil(bdims[1]/tile_size);
  //dim3 DimGrid1(bdims[0], bdims[3], w_grid*h_grid);
  dim3 DimGrid1(bdims[0], bdims[3], ceil(bdims[1]*bdims[2]/256.0));
  //dim3 DimBlock1(tile_size, tile_size, 1);
  dim3 DimBlock1(256, 1, 1);
  average_pool_kernel<<<DimGrid1, DimBlock1>>>(device_a, device_adims, pool_size, device_b, device_bdims, w_grid);

  cudaMemcpy(device_bdims, bdims, sizeof(int)*4, cudaMemcpyDeviceToHost);
  cudaMemcpy(b, device_b, sizeof(float)*device_b_size, cudaMemcpyDeviceToHost);
  // average pooling kernel end
  */


  // conv layer

  //const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
   //                    (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  //int device_c_size = cdims[0]*cdims[1]*cdims[2]*cdims[3];
  auto c = zeros<float>(cdims);
  //conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

  // kernel conv start
  // device_x holds device_b, device_conv1 holds device_conv2, device_a holds c

  cudaMemcpy(device_conv1, conv2, sizeof(float)*device_conv2_size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_conv1dims, conv2dims, sizeof(int)*4, cudaMemcpyHostToDevice);

  cudaMemcpy(device_adims, cdims, sizeof(int)*4, cudaMemcpyHostToDevice);
  cudaMemcpy(device_a, c, sizeof(float)*device_c_size, cudaMemcpyHostToDevice);

  //conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);
  dim3 DimGrid2(cdims[0], cdims[3], ceil(cdims[1]*cdims[2]/256.0));
  dim3 DimBlock2(256, 1, 1);

  const auto start3 = now();
  conv_forward_valid_kernel<<<DimGrid2, DimBlock2>>>(device_x, device_xdims, device_conv1, device_conv1dims, device_a, device_adims);
  const auto end3 = now();
  // cudaMemcpy(c, device_a, sizeof(float)*device_c_size, cudaMemcpyDeviceToHost);
  const auto elapsed3 = std::chrono::duration<double, std::milli>(end3 - start3).count();
  std::cout << "Done with " << bdims[0] << " queries in Conv2 "
            << "elapsed = " << elapsed3 << " milliseconds." << "\n";
  // kernel conv end

  // relu
  //relu4(c, cdims);

  // relu kernel start
  dim3 DimGrid3(ceil(device_a_size/256), 1, 1);
  dim3 DimBlock3(256, 1, 1);
  const auto start4 = now();
  relu4_kernel<<<DimGrid3, DimBlock3>>>(device_a, device_adims);
  // relu4(c, cdims);
  const auto end4 = now();
  const auto elapsed4 = std::chrono::duration<double, std::milli>(end4 - start4).count();
  std::cout << "Done with " << cdims[0] << " queries in relu4 "
            << "elapsed = " << elapsed4 << " milliseconds." << "\n";
  // cudaMemcpy(device_a, c, sizeof(float)*device_c_size, cudaMemcpyHostToDevice);
  // relu kernel end

  // average pooling
  // const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
  //                      cdims[3]};
  // int device_d_size = ddims[0]*ddims[1]*ddims[2]*ddims[3];
  auto d = zeros<float>(ddims);

  //pooling kernel start
  cudaMemcpy(device_xdims, ddims, sizeof(int)*4, cudaMemcpyHostToDevice);
  cudaMemcpy(device_x, d, sizeof(float)*device_d_size, cudaMemcpyHostToDevice);
  // w_grid = ceil(ddims[2]/tile_size);
  // h_grid = ceil(ddims[1]/tile_size);
  dim3 DimGrid4(ddims[0], ddims[3], ceil(ddims[1]*ddims[2]/256.0));
  dim3 DimBlock4(256, 1, 1);
  const auto start5 = now();
  average_pool_kernel<<<DimGrid4, DimBlock4>>>(device_a, device_adims, pool_size, device_x, device_xdims);//, w_grid);
  //average_pool(c, cdims, pool_size, d, ddims);
  const auto end5 = now();
  const auto elapsed5 = std::chrono::duration<double, std::milli>(end5 - start5).count();
  std::cout << "Done with " << ddims[0] << " queries in pool2 "
            << "elapsed = " << elapsed4 << " milliseconds." << "\n";
  // cudaMemcpy(d, device_x, sizeof(float)*device_d_size, cudaMemcpyDeviceToHost);
  //pooling kernel end

  // // reshape
  // const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

  // matrix multiplication
  // const int edims[] = {ddims[0], fc1dims[1]};
  auto e = zeros<float>(edims);

  // float* deviceD;
  // int* deviceDdims2;
  float* deviceFc1;
  int* deviceFc1dims;
  // float* deviceE;
  // int* deviceEdims;

  // int d2Size = ddims2[0] * ddims2[1];
  int fc1Size = fc1dims[0] * fc1dims[1];
  // int eSize = edims[0] * edims[1];

  cudaMalloc((void**) &deviceFc1, fc1Size * sizeof(float));
  cudaMalloc((void**) &deviceFc1dims, 2 * sizeof(int));

  // cudaMemcpy(device_x, d, d2Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_xdims, ddims2, 2 * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(device_conv1, fc1, fc1Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_conv1dims, fc1dims, 2 * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(device_adims, edims, 2 * sizeof(int), cudaMemcpyHostToDevice);

  dim3 DimGridE((edims[1]-1)/BLOCK_SIZE + 1, (edims[0]-1)/BLOCK_SIZE + 1, 1);
  dim3 DimBlockE(BLOCK_SIZE, BLOCK_SIZE, 1);
  // fully_forward(d, ddims2, fc1, fc1dims, e, edims);
  // get start time
  const auto start6 = now();
  fully_forward_kernel<<<DimGridE, DimBlockE>>>(device_x, device_xdims, device_conv1, device_conv1dims, device_a, device_adims);
  const auto end6 = now();
  const auto elapsed6 = std::chrono::duration<double, std::milli>(end6 - start6).count();
  std::cout << "Done with " << edims[0] << " queries in fully_forward "
           << "elapsed = " << elapsed6 << " milliseconds." << "\n";

  // cudaMemcpy(e, device_a, sizeof(float)*device_e_size, cudaMemcpyDeviceToHost);

  dim3 DimGridRelu2(ceil(device_e_size/256.0), 1, 1);
  dim3 DimBlockRelu2(256, 1, 1);
  const auto start7 = now();
  relu2_kernel<<<DimGridRelu2, DimBlockRelu2>>>(device_a, device_adims);
  // relu2(e, edims);
  const auto end7 = now();
  const auto elapsed7 = std::chrono::duration<double, std::milli>(end7 - start7).count();
  std::cout << "Done with " << edims[0] << " queries in relu2 "
           << "elapsed = " << elapsed7 << " milliseconds." << "\n";

  // matrix multiplication
  // const int fdims[] = {edims[0], fc2dims[1]};
  auto f = zeros<float>(fdims);

  // float* deviceFc2;
  // int* deviceFc2dims;
  // float* deviceF;
  // int* deviceFdims;

  int fc2size = fc2dims[0] * fc2dims[1];
  // int fsize = fdims[0] * fdims[1];
  // cudaMemcpy(device_a, e, device_e_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_conv1, fc2, fc2size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_conv1dims, fc2dims, 2 * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(device_xdims, fdims, 2 * sizeof(int), cudaMemcpyHostToDevice);

  dim3 DimGridF((fdims[1] - 1)/BLOCK_SIZE + 1, (fdims[0] - 1)/BLOCK_SIZE + 1, 1);
  dim3 DimBlockF(BLOCK_SIZE, BLOCK_SIZE, 1);
  const auto start8 = now();
  fully_forward_kernel<<<DimGridF, DimBlockF>>>(device_a, device_adims, device_conv1, device_conv1dims, device_x, device_xdims);
  //fully_forward(e, edims, fc2, fc2dims, f, fdims);
  const auto end8 = now();
  const auto elapsed8 = std::chrono::duration<double, std::milli>(end8 - start8).count();
  std::cout << "Done with " << fdims[0] << " queries in fully_forward 2 "
           << "elapsed = " << elapsed8 << " milliseconds." << "\n";

  cudaMemcpy(f, device_x, device_f_size * sizeof(float), cudaMemcpyDeviceToHost);

  // // matrix multiplication
  // const int edims[] = {ddims[0], fc1dims[1]};
  // auto e            = zeros<float>(edims);
  // fully_forward(d, ddims2, fc1, fc1dims, e, edims);
  //
  // // relu
  // relu2(e, edims);
  //
  // // matrix multiplication
  // const int fdims[] = {edims[0], fc2dims[1]};
  // auto f            = zeros<float>(fdims);
  // fully_forward(e, edims, fc2, fc2dims, f, fdims);

  // device_x_size = xdims[0]*xdims[1]*xdims[2]*xdims[3];
  // device_a_size = adims[0]*adims[1]*adims[2]*adims[3];
  // device_b_size = bdims[0]*bdims[1]*bdims[2]*bdims[3];
  // device_c_size = cdims[0]*cdims[1]*cdims[2]*cdims[3];
  // device_d_size = ddims[0]*ddims[1]*ddims[2]*ddims[3];
  // int device_d2_size = ddims2[0]*ddims2[1];
  // int device_e_size = edims[0]*edims[1];
  // int device_f_size = fdims[0]*fdims[1];

  // std::cout << "more devices: " << device_x_size << " " << device_a_size << " " << device_b_size << " " << device_c_size << " " << device_d_size << " " << device_d2_size << " " << device_e_size << " " << device_f_size << std::endl;
  // float* deviceF;
  // int* deviceFdims;
  // int* deviceOut;
  // float* deviceOutVal;
  //
  // cudaMalloc((void **) &deviceF, fsize * sizeof(float));
  // cudaMalloc((void **) &deviceFdims, 2 * sizeof(int));
  // cudaMalloc((void **) &deviceOut, fdims[0] * sizeof(int));
  // cudaMalloc((void **) &deviceOutVal, fdims[0] * sizeof(float));
  //
  // cudaMemcpy(deviceF, f, fsize * sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(deviceFdims, fdims, 2 * sizeof(int), cudaMemcpyHostToDevice);

  // dim3 DimGridArg(ceil(float)dims[1]/(BLOCK_SIZE*2)), ceil((float)fdims[0]/BLOCK_SIZE),1);
  // dim3 DimBlockArg(BLOCK_SIZE, BLOCK_SIZE,1);
  // dim3 DimGridArg(ceil((float)fdims[1]/(BLOCK_SIZE*2)), fdims[0],1);
  // dim3 DimBlockArg(BLOCK_SIZE, 1,1);

  const auto start9 = now();
  argmax(f, fdims, out);
  // argmax_kernel<<<DimGridArg, DimBlockArg>>>(deviceF, deviceFdims, deviceOutVal, deviceOut);
  const auto end9 = now();
  const auto elapsed9 = std::chrono::duration<double, std::milli>(end9 - start9).count();
  std::cout << "Done with argmax in " << "elapsed = " << elapsed << " milliseconds." << "\n";
  // std::cout<<"fdims: "<<fdims[0]<< " "<<fdims[1]<<"\n";

  // cudaMemcpy(out, deviceOut, fdims[0] * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(device_x);
  cudaFree(device_xdims);
  cudaFree(device_a);
  cudaFree(device_adims);
  cudaFree(device_conv1);
  cudaFree(device_conv1dims);
  cudaFree(deviceFc1);
  cudaFree(deviceFc1dims);

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
  // std::cout<<rdims[0]<<" "<<rdims[1]<<"\n";

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
