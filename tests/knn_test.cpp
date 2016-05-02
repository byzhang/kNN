/*
 * csv_test.cpp
 * Copyright (C) 2016- CloudBrain <byzhang@>
 */

#include "kNN/kNN.h"
#include <catch.hpp>
#include <glog/logging.h>
#include <vector>

using namespace std;

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

TEST_CASE("1M x 10", "[knn]") {
  const static uint32_t num_data = 1024 * 1024;
  const static uint32_t num_dim = 10;
  vector<uint32_t> data(num_data * num_dim);
  auto it = data.begin();
  for (uint32_t i = 0; i < num_data; ++i, it += num_dim) {
    int_to_vector(i, it, it + num_dim);
  }
  LOG(ERROR) << "init 1M x 10 data";

  kNN knn(data, num_data, num_dim);

  vector<uint32_t> query(num_dim);
  int_to_vector(0, query.begin(), query.end());
  auto results = knn.search(query, 100);
}
