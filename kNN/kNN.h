/*
 * kNN.h
 * Copyright (C) 2016 CloudBrain <byzhang@>
 */

#ifndef CloudBrain_KNN_H
#define CloudBrain_KNN_H

#include <memory>
#include <vector>
#include <functional>

class kNN {
 private:
  class impl;
  friend class kNN_Impl_CUB;
  friend class kNN_Impl_MGPU;
  std::unique_ptr<impl> self;
 
 public:
  kNN(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim, 
      std::function<impl* (const std::vector<uint32_t>&, uint32_t, uint32_t)> factory);
  virtual ~kNN();

  kNN(kNN&&);
  kNN& operator=(kNN&&);

  std::vector<uint32_t> search(const std::vector<uint32_t>& query, uint32_t top_k);
  
  template <class Impl>
  static impl* implFactory(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim);
};

template <class Impl> 
kNN::impl *kNN::implFactory(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim)
{
    return (new Impl(data, num_data, num_dim));
}

#endif /* !CloudBrain_KNN_H */
