#include "kNN_impl.h"
#include "distance.h"

#include <glog/logging.h>

using namespace std;

kNN::impl::~impl() {
  if (tex_) {
    cudaDestroyTextureObject(tex_);
  }
  if (data_) {
    cudaFreeArray(data_);
  }
}

kNN::impl::impl(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim)
  : num_data_(num_data), num_dim_(num_dim)
  , tex_height_(num_data / num_data_per_block + ((num_data % num_data_per_block)? 1: 0)) {
  CHECK(data.size() == num_data * num_dim) << "size mismatch:"
      << "data = " << data.size()
      << ", num_data * num_dim = " << num_data * num_dim;

  // allocate array
  auto channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
  auto error = cudaMallocArray(&data_, &channelDesc, num_dim * num_data_per_block, tex_height_);
  CHECK(error == cudaSuccess) << "error " << error << " when allocating data_ "
                              << "num_dim:" << num_dim_ << "; num_data:" << num_data_ << "; num_data_per_block:" << num_data_per_block
                              << "; tex_height:" << tex_height_;

  // memcpy array
  error = cudaMemcpyToArray(data_, 0, 0, data.data(), num_data_ * num_dim_ * sizeof(uint32_t), cudaMemcpyHostToDevice);
  CHECK(error == cudaSuccess) << "error " << error << " when copying to data_"
                              << "num_dim:" << num_dim_ << "; num_data:" << num_data_ << "; num_data_per_block:" << num_data_per_block
                              << "; tex_height:" << tex_height_;

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = data_;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture object
  error = cudaCreateTextureObject(&tex_, &resDesc, &texDesc, NULL);
  CHECK(error == cudaSuccess) << "error " << error << " when copying to data_"
                              << "num_dim:" << num_dim_ << "; num_data:" << num_data_ << "; num_data_per_block:" << num_data_per_block
                              << "; tex_height:" << tex_height_;
}
