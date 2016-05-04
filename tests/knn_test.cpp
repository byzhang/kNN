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
void int_to_vector(uint32_t d, IT begin, IT end) {
  size_t vector_length = distance(begin, end);
  uint32_t mask = 1;
  int r = 0;
  while (mask <= d) {
    int bit = (d & mask) / mask;
    if (bit) {
      int word_offset = r % vector_length;
      int bit_offset = r / vector_length;
      *(begin + word_offset) |= (1 << bit_offset);
    }
    ++r;
    mask <<= 1;
  }
}

TEST_CASE("0_to_vector", "[util]") {
  vector<uint32_t> data;
  data.resize(10);
  int_to_vector(0, data.begin(), data.end());
  for (size_t i = 0; i < data.size(); ++i) {
    INFO("i" << i);
    REQUIRE(data[i] == 0);
  }
}

TEST_CASE("1_to_vector", "[util]") {
  vector<uint32_t> data;
  data.resize(10);
  int_to_vector(1, data.begin(), data.end());
  for (size_t i = 0; i < data.size(); ++i) {
    INFO("i " << i);
    REQUIRE(data[i] == ((i == 0)? 1: 0));
  }
}

TEST_CASE("2_to_vector", "[util]") {
  vector<uint32_t> data;
  data.resize(10);
  int_to_vector(2, data.begin(), data.end());
  for (size_t i = 0; i < data.size(); ++i) {
    INFO("i " << i);
    REQUIRE(data[i] == ((i == 1)? 1: 0));
  }
}

TEST_CASE("7_to_vector", "[util]") {
  vector<uint32_t> data;
  data.resize(10);
  int_to_vector(7, data.begin(), data.end());
  for (size_t i = 0; i < 3; ++i) {
    INFO("i " << i);
    REQUIRE(data[i] == 1);
  }
  for (size_t i = 3; i < data.size(); ++i) {
    INFO("i " << i);
    REQUIRE(data[i] == 0);
  }
}

TEST_CASE("vector 1", "[util]") {
  for (uint32_t i = 0; i < 1000; ++i) {
    vector<uint32_t> data;
    data.resize(1);
    int_to_vector(i, data.begin(), data.end());
    INFO("i " << i);
    REQUIRE(data[0] == i);
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
