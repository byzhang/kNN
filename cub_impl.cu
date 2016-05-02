/*
 * cub_impl.cu
 * Copyright (C) 2016- CloudBrain <byzhang@>
 */

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include "cub_impl.h"
#include "distance.h"

#include <cub/cub.cuh>
#include <glog/logging.h>

using namespace cub;
using namespace std;

static const uint32_t num_data_per_block = 1024;
static cub::CachingDeviceAllocator allocator_;

kNN::impl::~impl() {
  if (tex_) {
    cudaDestroyTextureObject(tex_);
  }
  if (data_) {
    cudaFreeArray(data_);
  }
}

kNN::impl::impl(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim)
  : num_data_(num_data), num_dim_(num_dim)
  , tex_height_(num_data / num_data_per_block + ((num_data % num_data_per_block)? 1: 0)) {
  CHECK(data.size() == num_data * num_dim) << "size mismatch:"
      << "data = " << data.size()
      << ", num_data * num_dim = " << num_data * num_dim;

  // allocate array
  auto channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
  auto error = cudaMallocArray(&data_, &channelDesc, num_dim * num_data_per_block, tex_height_);
  CHECK(error == cudaSuccess) << "error " << error << " when allocating data_ "
                              << "num_dim:" << num_dim_ << "; num_data:" << num_data_ << "; num_data_per_block:" << num_data_per_block
                              << "; tex_height:" << tex_height_;

  // memcpy array
  error = cudaMemcpyToArray(data_, 0, 0, data.data(), num_data_ * num_dim_ * sizeof(uint32_t), cudaMemcpyHostToDevice);
  CHECK(error == cudaSuccess) << "error " << error << " when copying to data_"
                              << "num_dim:" << num_dim_ << "; num_data:" << num_data_ << "; num_data_per_block:" << num_data_per_block
                              << "; tex_height:" << tex_height_;

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = data_;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture object
  error = cudaCreateTextureObject(&tex_, &resDesc, &texDesc, NULL);
  CHECK(error == cudaSuccess) << "error " << error << " when copying to data_"
                              << "num_dim:" << num_dim_ << "; num_data:" << num_data_ << "; num_data_per_block:" << num_data_per_block
                              << "; tex_height:" << tex_height_;
}

#define CubDebugReturn(e, r) if (cub::Debug((e), __FILE__, __LINE__)) { return r; }

std::vector<uint32_t> kNN::impl::search(const std::vector<uint32_t>& query, uint32_t top_k) {
  std::vector<uint32_t> indexes{};
  if (query.size() != num_dim_) {
    LOG_EVERY_N(ERROR, 10000) << "size mismatch:"
                              << "query = " << query.size()
                              << ", num_dim = " << num_dim_;
    return indexes;
  }

  // TODO: thread local
  uint32_t* query_device = nullptr;
  auto error = allocator_.DeviceAllocate((void**)&query_device, sizeof(uint32_t) * num_dim_);
  if (error != cudaSuccess) {
    LOG_EVERY_N(ERROR, 1000) << "error " << error << " when aollcating query_device:" << num_dim_;
    return indexes;
  }

  CubDebugReturn(cudaMemcpy(query_device, query.data(), sizeof(uint32_t) * num_dim_, cudaMemcpyHostToDevice), indexes);

  // TODO: thread local
  DoubleBuffer<uint16_t> d_keys;
  DoubleBuffer<uint32_t> d_values;
  CubDebugReturn(allocator_.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(uint16_t) * num_data_), indexes);
  CubDebugReturn(allocator_.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(uint16_t) * num_data_), indexes);
  CubDebugReturn(allocator_.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(uint32_t) * num_data_), indexes);
  CubDebugReturn(allocator_.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(uint32_t) * num_data_), indexes);

  auto* keys = d_keys.d_buffers[0];
  auto* values = d_values.d_buffers[0];
  hamming_distance<<<tex_height_, num_data_per_block, num_dim_>>>(keys, values, query_device, tex_, tex_height_, num_dim_, num_data_per_block);

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;
  CubDebugReturn(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_data_), indexes);
  CubDebugReturn(allocator_.DeviceAllocate(&d_temp_storage, temp_storage_bytes), indexes);

  // Real sort
  d_keys.selector = d_values.selector = 0;
  CubDebugReturn(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_data_), indexes);
  cudaDeviceSynchronize();

  // copy to host
  indexes.resize(min(top_k, num_data_));
  CubDebugReturn(cudaMemcpy(indexes.data(), d_values.Current(), sizeof(uint32_t) * indexes.size(), cudaMemcpyDeviceToHost), indexes);

  return indexes;
}
