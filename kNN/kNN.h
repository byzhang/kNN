/*
 * kNN.h
 * Copyright (C) 2016 CloudBrain <byzhang@>
 */

#ifndef CloudBrain_KNN_H
#define CloudBrain_KNN_H

#include <memory>
#include <vector>

class kNN {
 public:
  kNN(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim);
  virtual ~kNN();

  kNN(kNN&&);
  kNN& operator=(kNN&&);

  std::vector<uint32_t> search(const std::vector<uint32_t>& query, uint32_t top_k);
 private:
  class impl;
  std::unique_ptr<impl> self;
};

#endif /* !CloudBrain_KNN_H */
