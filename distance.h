/*
 * distance.h
 * Copyright (C) 2016 CloudBrain <byzhang@>
 */

#ifndef CloudBrain_DISTANCE_H
#define CloudBrain_DISTANCE_H

#include <cuda_runtime.h>
#include <stdint.h>

// keys: distance
// values: index
__global__ void hamming_distance(uint16_t* keys, uint32_t *values, const uint32_t *query,
    const cudaTextureObject_t& tex, unsigned int tex_height, int num_dim, int num_data_per_block);

#endif /* !CloudBrain_DISTANCE_H */
