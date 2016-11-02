/*
 * kNN.cpp
 * Copyright (C) 2016- CloudBrain <byzhang@>
 */

#include "kNN/kNN.h"
#include "cub_impl.h"
#include "mgpu_impl.h"

#include <memory>

using namespace std;

kNN::kNN(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim,
         std::function<impl* (const std::vector<uint32_t>&, uint32_t, uint32_t)> factory)
  : self(unique_ptr<impl>(factory(data, num_data, num_dim))) {
}

std::vector<uint32_t> kNN::search(const std::vector<uint32_t>& query, uint32_t top_k) {
  return self->search(query, top_k);
}

kNN::~kNN() = default;
kNN::kNN(kNN&&) = default;
kNN& kNN::operator=(kNN&&) = default;
