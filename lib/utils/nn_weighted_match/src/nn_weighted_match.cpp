#include <torch/torch.h>
#include <iostream>
#include <vector>

extern THCState* state;

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace std;

void nn_weighted_match_launcher(
    at::Tensor descs0,  // B, N1, 2, D
    at::Tensor descs1,  // B, N2, 2, D
    at::Tensor idxs     // B, N1
);

void nn_weighted_match(
    at::Tensor descs0,  // B, N1, 2, D
    at::Tensor descs1,  // B, N2, 2, D
    at::Tensor idxs     // B, N1
) {
    CHECK_INPUT(descs0);
    CHECK_INPUT(descs1);
    CHECK_INPUT(idxs);

    nn_weighted_match_launcher(descs0, descs1, idxs);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nn_weighted_match", &nn_weighted_match, "nn weighted match");
}
