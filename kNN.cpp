/*
 * kNN.cpp
 * Copyright (C) 2016- CloudBrain <byzhang@>
 */

#include "kNN/kNN.h"
#include "cub_impl.h"

#include <memory>

using namespace std;

kNN::kNN(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim)
  : self{unique_ptr<kNN::impl>(new kNN::impl(data, num_data, num_dim))} {
}

std::vector<uint32_t> kNN::search(const std::vector<uint32_t>& query, uint32_t top_k) {
  return self->search(query, top_k);
}

kNN::~kNN() = default;
kNN::kNN(kNN&&) = default;
kNN& kNN::operator=(kNN&&) = default;
