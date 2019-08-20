#include "cuda_common.h"
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
//#include <ATen/Error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <curand_kernel.h>

__global__ void nn_weighted_match_kernel(
    float* descs0,  // B, N1, 2, D
    float* descs1,  // B, N2, 2, D
    int* idxs,      // B, N1
    int b, int n1, int d, int n2
) {

    // bni = bi * n1 + ni
    int bni = threadIdx.x + blockIdx.x * blockDim.x;
    if (bni >= b * n1)
        return;

    int bi = bni / n1;
    int n1i = bni - bi * n1;

    // the n1i-th element of descs0, i = bi * n1 * 2 * d + n1i * 2 * d
    float* desc0 = &descs0[bi*n1*2*d+n1i*2*d];
    int min_idx = -1;
    float min_dis = FLT_MAX;
    for (int n2i=0; n2i < n2; n2i++) {
        // the n2i-th element of descs1, i = bi * n2 * 2 * d + n2i * 2 * d
        float* desc1 = &descs1[bi*n2*2*d+n2i*2*d];

        // set2set min distance
        float set2set_min_dis = FLT_MAX;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                float dist = 0.f;
                for (int di=0; di < d; di++)
                    dist += (desc0[i*d+di] - desc1[j*d+di]) * (desc0[i*d+di] - desc1[j*d+di]);

                if (set2set_min_dis > dist)
                    set2set_min_dis = dist;
            }
        }

        if (min_dis > set2set_min_dis) {
            min_dis = set2set_min_dis;
            min_idx = n2i;
        }
    }

    // the n1i-th element of idxs, i = bi * n1 + n1i
    idxs[bi*n1+n1i] = min_idx;
}

void nn_weighted_match_launcher(
    at::Tensor descs0,  // B, N1, 2, D
    at::Tensor descs1,  // B, N2, 2, D
    at::Tensor idxs     // B, N1
) {
    int b = descs0.size(0);
    int n1 = descs0.size(1);
    int d = descs0.size(3);
    int n2 = descs1.size(1);

    assert(descs1.size(0) == b);
    assert(descs1.size(3) == d);
    assert(idxs.size(0) == b);
    assert(idxs.size(1) == n1);

    int bdim0, bdim1, bdim2;
    int tdim0, tdim1, tdim2;
    getGPULayout(b*n1, 1, 1, &bdim0, &bdim1, &bdim2, &tdim0, &tdim1, &tdim2);
    dim3 bdim(bdim0, bdim1, bdim2);
    dim3 tdim(tdim0, tdim1, tdim2);

    nn_weighted_match_kernel<<<bdim, tdim>>> (
        descs0.data<float>(),
        descs1.data<float>(),
        idxs.data<int>(),
        b, n1, d, n2
    );
    gpuErrchk(cudaGetLastError())
}
