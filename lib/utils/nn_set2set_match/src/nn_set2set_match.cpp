#include <torch/torch.h>
#include <iostream>
#include <vector>

extern THCState* state;

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace std;

void nn_linear_match_v1_launcher(
    at::Tensor descs0,  // B, N1, 2, D
    at::Tensor descs1,  // B, N2, 2, D
    at::Tensor idxs     // B, N1
);

void nn_linear_match_v1(
    at::Tensor descs0,  // B, N1, 2, D
    at::Tensor descs1,  // B, N2, 2, D
    at::Tensor idxs     // B, N1
) {
    CHECK_INPUT(descs0);
    CHECK_INPUT(descs1);
    CHECK_INPUT(idxs);

    nn_linear_match_v1_launcher(descs0, descs1, idxs);
}

void nn_linear_match_launcher(
    at::Tensor descs0,  // B, N1, 2, D
    at::Tensor descs1,  // B, N2, 2, D
    at::Tensor idxs     // B, N1
);

void nn_linear_match(
    at::Tensor descs0,  // B, N1, 2, D
    at::Tensor descs1,  // B, N2, 2, D
    at::Tensor idxs     // B, N1
) {
    CHECK_INPUT(descs0);
    CHECK_INPUT(descs1);
    CHECK_INPUT(idxs);

    nn_linear_match_launcher(descs0, descs1, idxs);
}

void nn_set2set_match_launcher(
    at::Tensor descs0,  // B, N1, 2, D
    at::Tensor descs1,  // B, N2, 2, D
    at::Tensor idxs     // B, N1
);

void nn_set2set_match(
    at::Tensor descs0,  // B, N1, 2, D
    at::Tensor descs1,  // B, N2, 2, D
    at::Tensor idxs     // B, N1
) {
    CHECK_INPUT(descs0);
    CHECK_INPUT(descs1);
    CHECK_INPUT(idxs);

    nn_set2set_match_launcher(descs0, descs1, idxs);
}

void nn_set2set_match_v1_launcher(
    at::Tensor descs0,  // B, N1, 2, D
    at::Tensor descs1,  // B, N2, 2, D
    at::Tensor idxs,    // B, N1
    at::Tensor scale_idxs  // B, N1
);

void nn_set2set_match_v1(
    at::Tensor descs0,  // B, N1, 2, D
    at::Tensor descs1,  // B, N2, 2, D
    at::Tensor idxs,    // B, N1
    at::Tensor scale_idxs  // B, N1
) {
    CHECK_INPUT(descs0);
    CHECK_INPUT(descs1);
    CHECK_INPUT(idxs);
    CHECK_INPUT(scale_idxs);

    nn_set2set_match_v1_launcher(descs0, descs1, idxs, scale_idxs);
}

void nn_match_launcher(
    at::Tensor descs0,  // B, N1, D
    at::Tensor descs1,  // B, N2, D
    at::Tensor idxs     // B, N1
);

void nn_match(
    at::Tensor descs0,  // B, N1, D
    at::Tensor descs1,  // B, N2, D
    at::Tensor idxs     // B, N1
) {
    CHECK_INPUT(descs0);
    CHECK_INPUT(descs1);
    CHECK_INPUT(idxs);

    nn_match_launcher(descs0, descs1, idxs);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nn_linear_match", &nn_linear_match, "nn linear match");
    m.def("nn_linear_match_v1", &nn_linear_match_v1, "nn linear match v1");
    m.def("nn_set2set_match", &nn_set2set_match, "nn set2set match");
    m.def("nn_set2set_match_v1", &nn_set2set_match_v1, "nn set2set match v1");
    m.def("nn_match", &nn_match, "nn match");
}
