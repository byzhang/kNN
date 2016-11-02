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

//static const uint32_t num_data_per_block = 1024;
static cub::CachingDeviceAllocator allocator_;

kNN_Impl_CUB::~kNN_Impl_CUB() {
}

kNN_Impl_CUB::kNN_Impl_CUB(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim)
  : kNN::impl(data, num_data, num_dim) {
}

#define CubDebugReturn(e, r) if (cub::Debug((e), __FILE__, __LINE__)) { return r; }

std::vector<uint32_t> kNN_Impl_CUB::search(const std::vector<uint32_t>& query, uint32_t top_k) {
  std::vector<uint32_t> indexes{};
  if (query.size() != num_dim_) {
    LOG_EVERY_N(ERROR, 10000) << "size mismatch:"
                              << "query = " << query.size()
                              << ", num_dim = " << num_dim_;
    return indexes;
  }

  // TODO: thread local
  uint32_t* d_query = nullptr;
  auto error = allocator_.DeviceAllocate((void**)&d_query, sizeof(uint32_t) * num_dim_);
  if (error != cudaSuccess) {
    LOG_EVERY_N(ERROR, 1000) << "error " << error << " when aollcating d_query:" << num_dim_;
    return indexes;
  }

  CubDebugReturn(cudaMemcpy(d_query, query.data(), sizeof(uint32_t) * num_dim_, cudaMemcpyHostToDevice), indexes);

  // TODO: thread local
  DoubleBuffer<KEY_T> d_keys;
  DoubleBuffer<uint32_t> d_values;
  CubDebugReturn(allocator_.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(KEY_T) * num_data_), indexes);
  CubDebugReturn(allocator_.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(KEY_T) * num_data_), indexes);
  CubDebugReturn(allocator_.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(uint32_t) * num_data_), indexes);
  CubDebugReturn(allocator_.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(uint32_t) * num_data_), indexes);

  auto* keys = d_keys.d_buffers[0];
  auto* values = d_values.d_buffers[0];
  hamming_distance<<<tex_height_, num_data_per_block, num_dim_ * sizeof(uint32_t)>>>(keys, values, d_query, tex_, tex_height_, num_dim_, num_data_per_block, num_data_);

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

  // cleanup
  if (d_keys.d_buffers[0]) CubDebugExit(allocator_.DeviceFree(d_keys.d_buffers[0]));
  if (d_keys.d_buffers[1]) CubDebugExit(allocator_.DeviceFree(d_keys.d_buffers[1]));
  if (d_values.d_buffers[0]) CubDebugExit(allocator_.DeviceFree(d_values.d_buffers[0]));
  if (d_values.d_buffers[1]) CubDebugExit(allocator_.DeviceFree(d_values.d_buffers[1]));
  if (d_temp_storage) CubDebugExit(allocator_.DeviceFree(d_temp_storage));
  if (d_query) CubDebugExit(allocator_.DeviceFree(d_query));

  return indexes;
}
