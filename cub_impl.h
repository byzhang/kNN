/*
 * cub_impl.h
 * Copyright (C) 2016 CloudBrain <byzhang@>
 */

#ifndef CloudBrain_CUB_IMPL_H
#define CloudBrain_CUB_IMPL_H

#include "kNN/kNN.h"

#include <cuda_runtime.h>
#include <vector>

class kNN::impl {
 public:
  impl(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim);
  virtual ~impl();
  std::vector<uint32_t> search(const std::vector<uint32_t>& query, uint32_t top_k);
 private:
  uint32_t num_data_;
  uint32_t num_dim_;
  uint32_t tex_height_;
  cudaArray* data_;
  cudaTextureObject_t tex_;
};


#endif /* !CloudBrain_CUB_IMPL_H */
