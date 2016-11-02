#ifndef CloudBrain_kNN_IMPL_H
#define CloudBrain_kNN_IMPL_H

#include "kNN/kNN.h"

#include <cuda_runtime.h>
#include <vector>


class kNN::impl {
 public:
     impl(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim);
     virtual ~impl();
     virtual std::vector<uint32_t> search(const std::vector<uint32_t>& query, uint32_t top_k) = 0;
 protected:
     const uint32_t num_data_per_block = 1024;
     uint32_t num_data_;
     uint32_t num_dim_;
     uint32_t tex_height_;
     cudaArray* data_;
     cudaTextureObject_t tex_;
};

#endif
