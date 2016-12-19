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

void unroll_w(float *w_in, float *w_out, int w0, int w1, int w2, int w3){
	for(int row=0; row<w3; row++){
		for(int p=0; p < w0; p++){
			for(int q=0; q<w1; q++){
				for(int c=0; c<w2; c++){
					int woffset = p * w1 * w2 * w3 + q * w2 * w3 + c*w3 + row;
					int unroll = row*w2*w1*w0 + c*w1*w0 + p*w0 + q;
					w_out[unroll] = w_in[woffset];
				}
			}
		}
	}
}

void unroll_x(float *x_in, float *x_out, int x0, int x1, int x2, int x3, int y1, int y2, int y3){
	for(int batch=0; batch<x0; batch++){
		for(int h=0; h<y1; h++){
			for(int w=0; w<y2; w++){
				// we fill x_unroll column by column
				for(int p=0; p<5; p++){
					for(int q=0; q<5; q++){
						for(int c=0; c<x3; c++){
							int h_unroll = p*5+q;
							int w_unroll = h*y2+w;
							int unroll_offset = batch*25*y1*y2+h_unroll*y1*y2 + w_unroll;
							int xoffset = batch*x1*x2 + (h+p)*x2 + (w+q);
							x_out[unroll_offset] = x_in[xoffset];
						}
					}
				}
			}
		}
	}
}

void unroll_mult(float *w_in, float *x_in, float *y_out, int x0, int x1, int x2, int x3, int y1, int y2, int y3){
	for(int batch=0; batch<x0; batch++){
		for(int row=0; row<y3; row++){
			for(int col=0; col<y1*y2; col++){
				float sum = 0;
				for(int inner = 0; inner<25; inner++){
					sum += w_in[row*25 + inner] * x_in[batch*25*y1*y2 + col + inner*y1*y2];
				}
				y_out[batch*y3*y1*y2 + row*y1*y2 + col] = sum;
			}
		}
	}
}

// y_in is the unrolled form
// y_out is the rerolled form
void reroll_y(float *y_in, float *y_out, int y0, int y1, int y2, int y3){
	for(int batch=0; batch<y0; batch++){
		for(int col=0; col<y2; col++){
			for(int row = 0; row<y1; row++){
				for(int img = 0; img<y3; img++){
					int yoffset = batch*y1*y2*y3 + row*y2*y3 + col*y3 + img;
					y_out[yoffset] = y_in[batch*y1*y2*y3 + row*y2 + col + img*y1*y2];
				}
			}
		}
	}
}
__global__ void unroll_w_kernel(float *w_in, float *w_out, int w0, int w1, int w2, int w3){
	int row = threadIdx.x + blockIdx.x*blockDim.x;
	if(row < w3){
	//for(int row=0; row<w3; row++){
		for(int p=0; p < w0; p++){
			for(int q=0; q<w1; q++){
				for(int c=0; c<w2; c++){
					int woffset = p * w1 * w2 * w3 + q * w2 * w3 + c*w3 + row;
					int unroll = row*w2*w1*w0 + c*w1*w0 + p*w0 + q;
					w_out[unroll] = w_in[woffset];
				}
			}
		}
	}
}

__global__ void unroll_x_kernel(float *x_in, float *x_out, int x0, int x1, int x2, int x3, int y1, int y2, int y3, int i){
	int batch = blockIdx.x;
	int col = threadIdx.x + i*96;
	int h = col/24;
	int w = col%24;
	//for(int batch=0; batch<x0; batch++){
		//for(int h=0; h<y1; h++){
			//for(int w=0; w<y2; w++){
				// we fill x_unroll column by column
				for(int p=0; p<5; p++){
					for(int q=0; q<5; q++){
						for(int c=0; c<x3; c++){
							int h_unroll = p*5+q;
							int w_unroll = h*y2+w;
							int unroll_offset = batch*25*y1*y2+h_unroll*y1*y2 + w_unroll;
							int xoffset = batch*x1*x2 + (h+p)*x2 + (w+q);
							x_out[unroll_offset] = x_in[xoffset];
						}
					}
				}
			//}
		//}
	//}
}

__global__ void unroll_mult_kernel(float *w_in, float *x_in, float *y_out, int i){
	int batch = blockIdx.x;
	int col = threadIdx.x + i*96;
	if(col < 24*24){
	//for(int batch=0; batch<10; batch++){
		for(int row=0; row<32; row++){
			//for(int col=0; col<24*24; col++){
				float sum = 0;
				for(int inner = 0; inner<25; inner++){
					sum += w_in[row*25 + inner] * x_in[batch*25*24*24 + col + inner*24*24];
				}
				y_out[batch*32*24*24 + row*24*24 + col] = sum;
			//}
		}
	//}
	}
}

// y_in is the unrolled form
// y_out is the rerolled form
__global__ void reroll_y_kernel(float *y_in, float *y_out, int y0, int y1, int y2, int y3){
	int batch = blockIdx.x;
	//for(int batch=0; batch<y0; batch++){
		for(int col=0; col<y2; col++){
			for(int row = 0; row<y1; row++){
				for(int img = 0; img<y3; img++){
					int yoffset = batch*y1*y2*y3 + row*y2*y3 + col*y3 + img;
					y_out[yoffset] = y_in[batch*y1*y2*y3 + row*y2 + col + img*y1*y2];
				}
			}
		}
	//}
}

// From book chapter Figure 16.4
static void conv_forward_valid(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
  const auto filter_h   = wdims[0];
  const auto filter_w   = wdims[1];
  const auto in_channel = wdims[2];

  std::cout << xdims[0] << " " << xdims[1] << " " << xdims[2] << " " <<xdims[3]<<std::endl;

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

void host_unroll(float *w_in, float *w_out, float *x_in, float *x_out, 
				 float *y_in, float *y_out, const int *wdims, const int *ydims){

  dim3 DimBlock(256, 1, 1);
  dim3 DimGrid(ceil(32*25/256.0), 1, 1);

  unroll_w_kernel<<<DimGrid, DimBlock>>>(w_in, w_out, wdims[0], wdims[1], wdims[2], wdims[3]);

  dim3 DimBlock1(96, 1, 1);
  dim3 DimGrid1(xdims[0], 1, 1);

  //for(int i=0; 
  for(int i=0; i<6; i++){
	  unroll_x_kernel<<<DimGrid1, DimBlock1>>> (x_in, x_out, xdims[0], xdims[1], xdims[2], xdims[3], ydims[1], ydims[2], ydims[3], i);
	  unroll_mult_kernel<<<DimGrid1, DimBlock1>>>(w_out, x_out, y_in, i);
  }

  dim3 DimBlock2(1, 1, 1);
  dim3 DimGrid2(xdims[0], 1, 1);
  reroll_y_kernel<<<DimGrid2, DimBlock2>>>(y_in, y_out, ydims[0], ydims[1], ydims[2], ydims[3]);
}

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  // conv layer
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  auto a = zeros<float>(adims);
  //conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);


  float *device_w;
  float *device_conv1;
  cudaMalloc((void **)&device_w, 32*25*sizeof(float));
  cudaMalloc((void **)&device_conv1, 32*25*sizeof(float));
  cudaMemcpy(device_conv1, conv1, 32*25 * sizeof(float), cudaMemcpyHostToDevice);

  float *device_x;
  float *d_unroll_x;
  int dxsize = xdims[0]* xdims[1]* xdims[2]* xdims[3];
  cudaMalloc((void **)&device_x, dxsize*sizeof(float));
  cudaMemcpy(device_x, x, dxsize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_unroll_x, xdims[0]*25*24*24*sizeof(float));

  float *device_a;
  cudaMalloc((void **)&device_a, adims[0]*adims[1]*adims[2]*adims[3]*sizeof(float));

  float *device_a_out;
  cudaMalloc((void **)&device_a_out, adims[0]*adims[1]*adims[2]*adims[3]*sizeof(float));

  auto s_w = zeros<float>(conv1dims[0] * conv1dims[1]*conv1dims[2]*conv1dims[3]);
  auto s_x = zeros<float>(adims[0]*adims[1]*adims[2]*25);
  auto s_y = zeros<float>(adims[0]*adims[1]*adims[2]*adims[3]);

  host_unroll(device_conv1, device_w, device_x, d_unroll_x, device_a, device_a_out,
				conv1dims, adims);

  cudaMemcpy(s_w, device_w, 32*25 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(s_x, d_unroll_x, xdims[0]*25*24*24 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(s_y, device_a, adims[0]*adims[1]*adims[2]*adims[3]* sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(a, device_a_out, adims[0]*adims[1]*adims[2]*adims[3] * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_x);
  cudaFree(device_w);
  cudaFree(d_unroll_x);
  cudaFree(device_a);
  cudaFree(device_conv1);
  cudaFree(device_a_out);

  delete[] s_w;
  delete[] s_x;
  delete[] s_y;


  /// relu layer
  relu4(a, adims);

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

  argmax(f, fdims, out);

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
