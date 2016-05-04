/*
 * csv_test.cpp
 * Copyright (C) 2016- CloudBrain <byzhang@>
 */

#include "kNN/kNN.h"

#include <atomic>
#include <catch.hpp>
#include <omp.h>
#include <glog/logging.h>
#include <vector>

using namespace std;

// begin should have at least 32/4 = 8 elements
template <typename IT>
void int_to_vector(uint32_t d, IT begin) {
  int mask = (1 << 4);
  for (IT it = begin + 32 / 4; it != begin;) {
    --it;
    *it = d % mask;
    d /= mask;
  }
}

template <typename IT>
string vector_to_string(IT begin, IT end) {
  ostringstream oss;
  for (auto it = begin; it != end; ++it) {
    oss << hex << *it;
  }
  return oss.str();
}

TEST_CASE("64M x 32", "[knn]") {
  const static uint32_t num_data = 64 * 1024 * 1024;
  const static uint32_t num_dim = 32;
  vector<uint32_t> data(num_data * num_dim);
  LOG(ERROR) << "generating test data";
  #pragma omp parallel for
  for (uint32_t i = 0; i < num_data; ++i) {
    int_to_vector(i, data.begin() + i * num_dim);
  }

  double start = omp_get_wtime();
  kNN knn(data, num_data, num_dim);
  double end = omp_get_wtime();
  double init_time = end - start;
  LOG(ERROR) << "init " << sizeof(uint32_t) * data.size() / 1024 / 1024 << "M bytes in " << init_time << "s";

  srand(time(nullptr));
  vector<uint32_t> queries(1024);
  for (size_t i = 0; i < queries.size(); ++i) {
    queries[i] = i + 1024;
    // queries[i] = rand() % num_data;
  }

  atomic_ullong query_time(0);
#define THREADS 1
  #pragma omp parallel for num_threads(THREADS)
  for (size_t i = 0; i < queries.size(); ++i) {
    vector<uint32_t> query(num_dim);
    int_to_vector(queries[i], query.begin());
    double start = omp_get_wtime();
    auto results = knn.search(query, 100);
    query_time += (omp_get_wtime() - start) * 1000000;
    vector<uint16_t> distance(results.size());
    for (size_t j = 0; j < results.size(); ++j) {
      distance[j] = __builtin_popcount(results[j] ^ queries[i]);
      INFO("i:" << i << " j:" << j << " query:" << queries[i] << " result:" << results[j] << " distance:" << distance[j]
           << " input:" << vector_to_string(query.begin(), query.end())
           << " data:" << vector_to_string(data.begin() + results[j] * num_dim, data.begin() + results[j] * num_dim + num_dim));
      if (j == 0) {
        REQUIRE(results[j] == queries[i]);
      } else {
        REQUIRE(distance[j] >= distance[j - 1]);
      }
    }
  }
  LOG(ERROR) << "search top 100 NN for " << num_dim << "d queries in " << query_time / queries.size() / 1000000.0 / THREADS << "s";
}
