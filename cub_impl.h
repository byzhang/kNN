/*
 * cub_impl.h
 * Copyright (C) 2016 CloudBrain <byzhang@>
 */

#ifndef CloudBrain_CUB_IMPL_H
#define CloudBrain_CUB_IMPL_H

#include "kNN/kNN_impl.h"

#include <cuda_runtime.h>
#include <vector>

class kNN_Impl_CUB: public kNN::impl {
 public:
     kNN_Impl_CUB(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim);
     ~kNN_Impl_CUB() override;
     std::vector<uint32_t> search(const std::vector<uint32_t>& query, uint32_t top_k) override;
};


#endif /* !CloudBrain_CUB_IMPL_H */
