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

__global__ void conv_forward_valid_kernel(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
  const auto filter_h   = wdims[0];
  const auto filter_w   = wdims[1];
  const auto in_channel = wdims[2];
  const auto n_out = ydims[0];
  const auto m_out = ydims[3];
  const auto w_out = ydims[2];
  const auto h_out = ydims[1];
  //const int w_grid = ceil(w_out/16.0);

  int n, m , h, w;

  n = blockIdx.x;
  m = blockIdx.y;
  h = blockIdx.z/w_out + threadIdx.y;
  w = blockIdx.z%w_out + threadIdx.x;

  if(n < n_out && m < m_out && h < h_out && w < w_out){
    float acc = 0;

    for (const auto p : range(0, filter_h)) {
      for (const auto q : range(0, filter_w)) {
        for (const auto c : range(0, in_channel)) {
          const auto xoffset = n * xdims[1] * xdims[2] * xdims[3] +
                               (h + p) * xdims[2] * xdims[3] +
                               (w + q) * xdims[3] + c;
          const auto woffset = p * wdims[1] * wdims[2] * wdims[3] +
                               q * wdims[2] * wdims[3] + c * wdims[3] + m;
          std::cout<<X[xoffset]<<" ";
          acc += X[xoffset]*W[woffset];
         }
       }
     }

     const auto yoffset =
         ((n * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
     Y[yoffset] = acc;
  }

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

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  // conv layer
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  auto a = zeros<float>(adims);

  float* g1 = allocate<float>(xdims);
  float* g2 = allocate<float>(conv1dims);

  float* deviceX;
  float* deviceConv1;
  float* deviceA;

  std::cout<<"x prior"<<std::endl;
  for(int i = 0; i < xdims[0]*xdims[1]*xdims[2]*xdims[3]; i++) {
    std::cout<<x[i]<<' ';
  }

  cudaMalloc((void**) &deviceX, xdims[0]*xdims[1]*xdims[2]*xdims[3] * sizeof(float));
  cudaMalloc((void**) &deviceConv1, conv1dims[0]*conv1dims[1]*conv1dims[2]*conv1dims[3] * sizeof(float));
  cudaMalloc((void**) &deviceA, adims[0]*adims[1]*adims[2]*adims[3] * sizeof(float));

  cudaMemcpy(deviceX, x, xdims[0]*xdims[1]*xdims[2]*xdims[3] * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceConv1, conv1, conv1dims[0]*conv1dims[1]*conv1dims[2]*conv1dims[3] * sizeof(float), cudaMemcpyHostToDevice);

  // cudaMemcpy(g1, deviceX, xdims[0]*xdims[1]*xdims[2]*xdims[3] * sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(g2, deviceConv1, conv1dims[0]*conv1dims[1]*conv1dims[2]*conv1dims[3] * sizeof(float), cudaMemcpyDeviceToHost);
  //
  // std::cout<<"\nx"<<std::endl;
  // for(int i = 0; i < xdims[0]*xdims[1]*xdims[2]*xdims[3]; i++) {
  //   std::cout<<x[i]<<' ';
  // }
  //
  // std::cout<<"\ng1"<<std::endl;
  // for(int i = 0; i < xdims[0]*xdims[1]*xdims[2]*xdims[3]; i++) {
  //   std::cout<<g1[i]<<' ';
  // }

  const int W_grid = ceil(adims[2]/16.0);
  const int H_grid = ceil(adims[1]/16.0);
  const int Z = W_grid*H_grid;
  const int N = adims[0];
  const int M = adims[3];

  dim3 convBlockDim(16, 16, 1);
  dim3 convGridDim(N, M, Z);

  // get start time
  const auto start = now();

  conv_forward_valid_kernel<<<convGridDim, convBlockDim>>>(deviceX, xdims, deviceConv1, conv1dims, deviceA, adims);
  //conv_forward_valid(g1, xdims, g2, conv1dims, a, adims);

  // get end time
  const auto end = now();

  // cudaMemcpy(a, deviceA, adims[0]*adims[1]*adims[2]*adims[3] * sizeof(float), cudaMemcpyDeviceToHost); //we won't need to copy over in future
  // cudaFree(deviceX);
  // cudaFree(deviceConv1);
  // cudaFree(deviceA); // When we're set we won't free this until after avg pooling

  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "Done with " << adims[0] << " queries in Conv1 "
            << "elapsed = " << elapsed << " milliseconds."
            << "\n";

  // get start time
  const auto start1 = now();

  /// relu layer
  relu4(a, adims);

  // get end time
  const auto end1 = now();

  const auto elapsed1 =
      std::chrono::duration<double, std::milli>(end1 - start1).count();

  std::cout << "Done with " << adims[0] << " queries in relu4 "
            << "elapsed = " << elapsed1 << " milliseconds."
            << "\n";

  // average pooling
  const int pool_size = 2;
  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                       adims[3]};
  auto b = zeros<float>(bdims);

  // get start time
  const auto start2 = now();

  average_pool(a, adims, pool_size, b, bdims);

  // get end time
  const auto end2 = now();

  const auto elapsed2 =
      std::chrono::duration<double, std::milli>(end2 - start2).count();

  std::cout << "Done with " << adims[0] << " queries in pool1 "
            << "elapsed = " << elapsed2 << " milliseconds."
            << "\n";

  // conv layer
  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
                       (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  auto c = zeros<float>(cdims);

  // get start time
  const auto start3 = now();

  conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

  // get end time
  const auto end3 = now();

  const auto elapsed3 =
      std::chrono::duration<double, std::milli>(end3 - start3).count();

  std::cout << "Done with " << bdims[0] << " queries in Conv2 "
            << "elapsed = " << elapsed3 << " milliseconds."
            << "\n";

  // get start time
  const auto start4 = now();

  // relu
  relu4(c, cdims);

  // get end time
  const auto end4 = now();

  const auto elapsed4 =
      std::chrono::duration<double, std::milli>(end4 - start4).count();

  std::cout << "Done with " << cdims[0] << " queries in relu4 "
            << "elapsed = " << elapsed4 << " milliseconds."
            << "\n";

  // average pooling
  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                       cdims[3]};
  auto d = zeros<float>(ddims);

  // get start time
  const auto start5 = now();

  average_pool(c, cdims, pool_size, d, ddims);

  // get end time
  const auto end5 = now();

  const auto elapsed5 =
      std::chrono::duration<double, std::milli>(end5 - start5).count();

  std::cout << "Done with " << cdims[0] << " queries in pool2 "
            << "elapsed = " << elapsed5 << " milliseconds."
            << "\n";

  // reshape
  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

  // matrix multiplication
  const int edims[] = {ddims[0], fc1dims[1]};
  auto e            = zeros<float>(edims);

  // get start time
  const auto start6 = now();

  fully_forward(d, ddims2, fc1, fc1dims, e, edims);

  // get end time
  const auto end6 = now();

  const auto elapsed6 =
      std::chrono::duration<double, std::milli>(end6 - start6).count();

  std::cout << "Done with " << ddims2[0] << " queries in mm1 "
            << "elapsed = " << elapsed6 << " milliseconds."
            << "\n";

  // get start time
  const auto start7 = now();

  // relu
  relu2(e, edims);

  // get end time
  const auto end7 = now();

  const auto elapsed7 =
      std::chrono::duration<double, std::milli>(end7 - start7).count();

  std::cout << "Done with " << edims[0] << " queries in relu2 "
            << "elapsed = " << elapsed7 << " milliseconds."
            << "\n";

  // matrix multiplication
  const int fdims[] = {edims[0], fc2dims[1]};
  auto f            = zeros<float>(fdims);

  // get start time
  const auto start8 = now();

  fully_forward(e, edims, fc2, fc2dims, f, fdims);

  // get end time
  const auto end8 = now();

  const auto elapsed8 =
      std::chrono::duration<double, std::milli>(end8 - start8).count();

  std::cout << "Done with " << edims[0] << " queries in mm2 "
            << "elapsed = " << elapsed8 << " milliseconds."
            << "\n";

  // get start time
  const auto start9 = now();

  argmax(f, fdims, out);

  // get end time
  const auto end9 = now();

  const auto elapsed9 =
      std::chrono::duration<double, std::milli>(end9 - start9).count();

  std::cout << "Done with " << edims[0] << " queries in argmax "
            << "elapsed = " << elapsed9 << " milliseconds."
            << "\n";

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
