#include "mgpu_impl.h"
#include "distance.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <moderngpu/kernel_mergesort.hxx>
#include <glog/logging.h>

kNN_Impl_MGPU::kNN_Impl_MGPU(const std::vector<uint32_t>& data, uint32_t num_data, uint32_t num_dim)
: kNN::impl(data, num_data, num_dim)
{
}

kNN_Impl_MGPU::~kNN_Impl_MGPU()
{
}

//potential mem leak
#define CudaDebugReturn(e, r) if (cudaSuccess != e) { return r; }

std::vector<uint32_t> kNN_Impl_MGPU::search(const std::vector<uint32_t>& query, uint32_t top_k)
{
    static mgpu::standard_context_t context;
    
    std::vector<uint32_t> indexes{};
    if (query.size() != num_dim_) {
      LOG_EVERY_N(ERROR, 10000) << "size mismatch:"
                                << "query = " << query.size()
                                << ", num_dim = " << num_dim_;
      return indexes;
    }

    // TODO: thread local
    uint32_t* d_query = nullptr;
    auto error = cudaMalloc(&d_query, sizeof(uint32_t) * num_dim_);
    if (error != cudaSuccess) {
      LOG_EVERY_N(ERROR, 1000) << "error " << error << " when aollcating d_query:" << num_dim_;
      return indexes;
    }

    CudaDebugReturn(cudaMemcpy(d_query, query.data(), sizeof(uint32_t) * num_dim_, cudaMemcpyHostToDevice), indexes);

    KEY_T* d_keys = nullptr;
    uint32_t* d_values = nullptr;
    
    error = cudaMalloc(&d_keys, sizeof(KEY_T) * num_data_);
    if (cudaSuccess != error)
    {
        LOG_EVERY_N(ERROR, 1000) << "error " << error << " when aollcating d_keys:" << num_data_;
        return indexes;
    }

    error = cudaMalloc(&d_values, sizeof(uint32_t) * num_data_);
    if (cudaSuccess != error)
    {
        LOG_EVERY_N(ERROR, 1000) << "error " << error << " when aollcating d_values:" << num_data_;
        return indexes;
    }
    
    hamming_distance<<<tex_height_, num_data_per_block, num_dim_ * sizeof(uint32_t)>>>(d_keys, d_values, d_query, tex_, tex_height_, num_dim_, num_data_per_block);

    mgpu::mergesort(d_keys, d_values, num_data_, mgpu::less_t<KEY_T>(), context);
    context.synchronize();
    // copy to host
    indexes.resize(min(top_k, num_data_));
    CudaDebugReturn(cudaMemcpy(indexes.data(), d_values, sizeof(uint32_t) * indexes.size(), cudaMemcpyDeviceToHost), indexes);

    // cleanup
    if (d_query) CudaDebugReturn(cudaFree(d_query), indexes);
    if (d_keys) CudaDebugReturn(cudaFree(d_keys), indexes);
    if (d_values) CudaDebugReturn(cudaFree(d_values), indexes);

    return indexes;

}
