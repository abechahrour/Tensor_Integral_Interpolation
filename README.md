# TI_ML_Interpolation

Interpolating Tensor Integrals through neural nets for light-by-light scattering.

Tensor Integrals computed through GOLEM (https://arxiv.org/abs/0810.0992).

Coefficients of tensor integrals computed through GOLEM via Tensorial Reconstruction (https://arxiv.org/abs/1008.2441).

## The idea is as follows:
1. Choose n-point rank-r tensor integral
2. Generate training data by computing numerical value over some phase space using GOLEM
3. Train NN for the tensor integral
4. When computing the matrix element, use GOLEM for coefficients and NN for tensor integral
