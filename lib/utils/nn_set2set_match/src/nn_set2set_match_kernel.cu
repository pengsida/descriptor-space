#include "cuda_common.h"
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
//#include <ATen/Error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <curand_kernel.h>

__global__ void nn_set2set_match_kernel(
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

    int level = 2;

    // the n1i-th element of descs0, i = bi * n1 * level * d + n1i * level * d
    float* desc0 = &descs0[bi*n1*level*d+n1i*level*d];
    int min_idx = -1;
    float min_dis = FLT_MAX;
    for (int n2i=0; n2i < n2; n2i++) {
        // the n2i-th element of descs1, i = bi * n2 * level * d + n2i * level * d
        float* desc1 = &descs1[bi*n2*level*d+n2i*level*d];

        // set2set min distance
        float set2set_min_dis = FLT_MAX;
        for (int i = 0; i < level; i++) {
            float dist = 0.f;
            for (int di=0; di < d; di++)
                dist += (desc0[i*d+di] - desc1[di]) * (desc0[i*d+di] - desc1[di]);
            if (set2set_min_dis > dist)
                set2set_min_dis = dist;
        }
        for (int i = 1; i < level; i++) {
            float dist = 0.f;
            for (int di=0; di < d; di++)
                dist += (desc0[di] - desc1[i*d+di]) * (desc0[di] - desc1[i*d+di]);
            if (set2set_min_dis > dist)
                set2set_min_dis = dist;
        }

        // for (int i = 0; i < 2; i++) {
        //     for (int j = 0; j < 2; j++) {
        //         if (i == 1 && j == 1)
        //             continue;
        //
        //         float dist = 0.f;
        //         for (int di=0; di < d; di++)
        //             dist += (desc0[i*d+di] - desc1[j*d+di]) * (desc0[i*d+di] - desc1[j*d+di]);

        //         if (set2set_min_dis > dist)
        //             set2set_min_dis = dist;
        //     }
        // }

        if (min_dis > set2set_min_dis) {
            min_dis = set2set_min_dis;
            min_idx = n2i;
        }
    }

    // the n1i-th element of idxs, i = bi * n1 + n1i
    idxs[bi*n1+n1i] = min_idx;
}

void nn_set2set_match_launcher(
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

    nn_set2set_match_kernel<<<bdim, tdim>>> (
        descs0.data<float>(),
        descs1.data<float>(),
        idxs.data<int>(),
        b, n1, d, n2
    );
    gpuErrchk(cudaGetLastError())
}

__global__ void nn_set2set_match_v1_kernel(
    float* descs0,  // B, N1, 2, D
    float* descs1,  // B, N2, 2, D
    int* idxs,      // B, N1
    int* scale_idxs,  // B, N1
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
    int min_scale_idx = -1;
    float min_dis = FLT_MAX;
    for (int n2i=0; n2i < n2; n2i++) {
        // the n2i-th element of descs1, i = bi * n2 * 2 * d + n2i * 2 * d
        float* desc1 = &descs1[bi*n2*2*d+n2i*2*d];

        // set2set min distance
        int set2set_min_scale_idx = -1;
        float set2set_min_dis = FLT_MAX;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                if (i == 1 && j == 1)
                    continue;

                float dist = 0.f;
                for (int di=0; di < d; di++)
                    dist += (desc0[i*d+di] - desc1[j*d+di]) * (desc0[i*d+di] - desc1[j*d+di]);

                if (set2set_min_dis > dist) {
                    set2set_min_dis = dist;
                    set2set_min_scale_idx = i * 2 + j;
                }
            }
        }

        if (min_dis > set2set_min_dis) {
            min_dis = set2set_min_dis;
            min_idx = n2i;
            min_scale_idx = set2set_min_scale_idx;
        }
    }

    // the n1i-th element of idxs, i = bi * n1 + n1i
    idxs[bi*n1+n1i] = min_idx;
    scale_idxs[bi*n1+n1i] = min_scale_idx;
}

void nn_set2set_match_v1_launcher(
    at::Tensor descs0,  // B, N1, 2, D
    at::Tensor descs1,  // B, N2, 2, D
    at::Tensor idxs,    // B, N1
    at::Tensor scale_idxs  // B, N1
) {
    int b = descs0.size(0);
    int n1 = descs0.size(1);
    int d = descs0.size(3);
    int n2 = descs1.size(1);

    assert(descs1.size(0) == b);
    assert(descs1.size(3) == d);
    assert(idxs.size(0) == b);
    assert(idxs.size(1) == n1);
    assert(scale_idxs.size(0) == b);
    assert(scale_idxs.size(1) == n1);

    int bdim0, bdim1, bdim2;
    int tdim0, tdim1, tdim2;
    getGPULayout(b*n1, 1, 1, &bdim0, &bdim1, &bdim2, &tdim0, &tdim1, &tdim2);
    dim3 bdim(bdim0, bdim1, bdim2);
    dim3 tdim(tdim0, tdim1, tdim2);

    nn_set2set_match_v1_kernel<<<bdim, tdim>>> (
        descs0.data<float>(),
        descs1.data<float>(),
        idxs.data<int>(),
        scale_idxs.data<int>(),
        b, n1, d, n2
    );
    gpuErrchk(cudaGetLastError())
}

__global__ void nn_match_kernel(
    float* descs0,  // B, N1, D
    float* descs1,  // B, N2, D
    int* idxs,      // B, N1
    int b, int n1, int d, int n2
) {
    // bni = bi * n1 + ni
    int bni = threadIdx.x + blockIdx.x * blockDim.x;
    if (bni >= b * n1)
        return;

    int bi = bni / n1;
    int n1i = bni - bi * n1;

    // the n1i-th element of descs0, i = bi * n1 * d + n1i * d
    float* desc0 = &descs0[bi*n1*d+n1i*d];
    int min_idx = -1;
    float min_dis = FLT_MAX;
    for (int n2i=0; n2i < n2; n2i++) {
        // the n2i-th element of descs1, i = bi * n2 * d + n2i * d
        float* desc1 = &descs1[bi*n2*d+n2i*d];
        float dist = 0.f;
        for (int di=0; di < d; di++)
            dist += (desc0[di] - desc1[di]) * (desc0[di] - desc1[di]);

        if (min_dis > dist) {
            min_dis = dist;
            min_idx = n2i;
        }
    }

    // the n1i-th element of idxs, i = bi * n1 + n1i
    idxs[bi*n1+n1i] = min_idx;
}

void nn_match_launcher(
    at::Tensor descs0,  // B, N1, D
    at::Tensor descs1,  // B, N2, D
    at::Tensor idxs     // B, N1
) {
    int b = descs0.size(0);
    int n1 = descs0.size(1);
    int d = descs0.size(2);
    int n2 = descs1.size(1);

    assert(descs1.size(0) == b);
    assert(descs1.size(2) == d);
    assert(idxs.size(0) == b);
    assert(idxs.size(1) == n1);

    int bdim0, bdim1, bdim2;
    int tdim0, tdim1, tdim2;
    getGPULayout(b*n1, 1, 1, &bdim0, &bdim1, &bdim2, &tdim0, &tdim1, &tdim2);
    dim3 bdim(bdim0, bdim1, bdim2);
    dim3 tdim(tdim0, tdim1, tdim2);

    nn_match_kernel<<<bdim, tdim>>> (
        descs0.data<float>(),
        descs1.data<float>(),
        idxs.data<int>(),
        b, n1, d, n2
    );
    gpuErrchk(cudaGetLastError())
}


__global__ void nn_linear_match_kernel(
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

    int level = 2;

    // the n1i-th element of descs0, i = bi * n1 * level * d + n1i * level * d
    // desc0: (2, D)
    float* desc0 = &descs0[bi*n1*level*d+n1i*level*d];
    int min_idx = -1;
    float min_dis = FLT_MAX;


    for (int n2i=0; n2i < n2; n2i++) {
        // the n2i-th element of descs1, i = bi * n2 * level * d + n2i * level * d
        // desc0: desc1: (2, D)
        float* desc1 = &descs1[bi*n2*level*d+n2i*level*d];

        // compute weight
        // compute A00, A01, A10, A11
        float d0;
        float d1;
        float A[4][4] = {0};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < d; k++) {
                    if (i == 0 || i == 1) {
                        d0 = desc0[i * d + k];
                    }
                    else {
                        d0 = -desc1[(i - 2) * d + k];
                    }
                    if (j == 0 || j == 1) {
                        d1 = desc0[j * d + k];
                    }
                    else {
                        d1 = -desc1[(j - 2) * d + k];
                    }
                    A[i][j] += d0 * d1;
                }
            }
        }
        // find minimum
        float a, b;
        float x0, x1, x2, x3;
        x0 = x1 = x2 = x3 = 0.5;

        for (int round = 0; round < 3; round ++) {
            a = A[0][0]  + A[1][1]  - 2 * A[0][1];
            b = 2 * (A[0][1] - A[1][1] +
                     x2 * (A[0][2] - A[1][2]) + x3 * (A[0][3] - A[1][3]));

            x0 = - b / (2 * a);
            if (x0 < 0.0) x0 = 0.0;
            if (x0 > 1.0) x0 = 1.0;
            x1 = 1 - x0;

            // optimize x2, x3
            a = A[2][2] + A[3][3]  - 2 * A[2][3];
            b = 2 * (A[2][3] - A[3][3]  +
                     x0 * (A[0][2] - A[0][3]) + x1 * (A[1][2] - A[1][3]));

            x2 = - b / (2 * a);
            if (x2 < 0.0) x2 = 0.0;
            if (x2 > 1.0) x2 = 1.0;
            x3 = 1 - x2;

        }

        // now x0, x1, x2, x3 are ready
        // linear distance
        float dist = 0.f;
        float diff = 0.f;
        for (int di=0; di < d; di++) {
            // dist += (desc0[di] - desc1[i*d+di]) * (desc0[di] - desc1[i*d+di]);
            diff = (x0 * desc0[di] + x1 * desc0[d + di] - x2 * desc1[di] - x3 * desc1[d + di]);
            dist += diff * diff;
        }
        if (min_dis > dist) {
            min_dis = dist;
            min_idx = n2i;
        }
    }

    // the n1i-th element of idxs, i = bi * n1 + n1i
    idxs[bi*n1+n1i] = min_idx;
}

void nn_linear_match_launcher(
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

    nn_linear_match_kernel<<<bdim, tdim>>> (
        descs0.data<float>(),
        descs1.data<float>(),
        idxs.data<int>(),
        b, n1, d, n2
    );
    gpuErrchk(cudaGetLastError())
}

__global__ void nn_linear_match_v1_kernel(
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

    int level = 2;

    // the n1i-th element of descs0, i = bi * n1 * level * d + n1i * level * d
    // desc0: (2, D)
    float* desc0 = &descs0[bi*n1*level*d+n1i*level*d];
    int min_idx = -1;
    float min_dis = FLT_MAX;


    for (int n2i=0; n2i < n2; n2i++) {
        // the n2i-th element of descs1, i = bi * n2 * level * d + n2i * level * d
        // desc0: desc1: (2, D)
        float* desc1 = &descs1[bi*n2*level*d+n2i*level*d];

        // compute weight
        // compute A00, A01, A10, A11
        float d0;
        float d1;
        float A[4][4] = {0};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < d; k++) {
                    if (i == 0 || i == 1) {
                        d0 = desc0[i * d + k];
                    }
                    else {
                        d0 = -desc1[(i - 2) * d + k];
                    }
                    if (j == 0 || j == 1) {
                        d1 = desc0[j * d + k];
                    }
                    else {
                        d1 = -desc1[(j - 2) * d + k];
                    }
                    A[i][j] += d0 * d1;
                }
            }
        }

        // find minimum
        float a, b;
        float x0, x1, x2, x3;

        // fix weight for desc0
        x0 = 1; x1 = 0;
        x2 = x3 = 0.5;
        // optimize x2, x3
        a = A[2][2] + A[3][3]  - 2 * A[2][3];
        b = 2 * (A[2][3] - A[3][3]  +
                 x0 * (A[0][2] - A[0][3]) + x1 * (A[1][2] - A[1][3]));

        x2 = - b / (2 * a);
        if (x2 < 0.0) x2 = 0.0;
        if (x2 > 1.0) x2 = 1.0;
        x3 = 1 - x2;

        // now x0, x1, x2, x3 are ready
        // linear distance
        float dist = 0.f;
        float diff = 0.f;
        for (int di=0; di < d; di++) {
            // dist += (desc0[di] - desc1[i*d+di]) * (desc0[di] - desc1[i*d+di]);
            diff = (x0 * desc0[di] + x1 * desc0[d + di] - x2 * desc1[di] - x3 * desc1[d + di]);
            dist += diff * diff;
        }
        if (min_dis > dist) {
            min_dis = dist;
            min_idx = n2i;
        }


        // fix weight for desc0
        x0 = 0.5; x1 = 0.5;
        x2 = 1; x3 = 0;
        // compute distance

        a = A[0][0]  + A[1][1]  - 2 * A[0][1];
        b = 2 * (A[0][1] - A[1][1] +
                 x2 * (A[0][2] - A[1][2]) + x3 * (A[0][3] - A[1][3]));

        x0 = - b / (2 * a);
        if (x0 < 0.0) x0 = 0.0;
        if (x0 > 1.0) x0 = 1.0;
        x1 = 1 - x0;

        // now x0, x1, x2, x3 are ready
        // linear distance
        dist = 0.f;
        diff = 0.f;
        for (int di=0; di < d; di++) {
            // dist += (desc0[di] - desc1[i*d+di]) * (desc0[di] - desc1[i*d+di]);
            diff = (x0 * desc0[di] + x1 * desc0[d + di] - x2 * desc1[di] - x3 * desc1[d + di]);
            dist += diff * diff;
        }
        if (min_dis > dist) {
            min_dis = dist;
            min_idx = n2i;
        }
    }

    // the n1i-th element of idxs, i = bi * n1 + n1i
    idxs[bi*n1+n1i] = min_idx;
}

void nn_linear_match_v1_launcher(
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

    nn_linear_match_kernel<<<bdim, tdim>>> (
        descs0.data<float>(),
        descs1.data<float>(),
        idxs.data<int>(),
        b, n1, d, n2
    );
    gpuErrchk(cudaGetLastError())
}
