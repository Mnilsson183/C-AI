#ifndef TENSOR_H
#define TENSOR_H

#include <sys/_types/_u_int32_t.h>
#include <sys/_types/_u_int64_t.h>

typedef u_int32_t u32;
typedef u_int64_t u64;

typedef struct {
	u32* shape;
	u32 rank;
	u64 size;
	double* data;
} Tensor;

Tensor* create_tensor_zeros(u32 rank, u32* shape);
Tensor* create_tensor_random(u32 rank, u32* shape);
Tensor* create_tensor_array(double* source, u32 rank, u32* shape);
void free_tensor(Tensor* tensor);

float get_element(Tensor* tensor, u32* indices);
void set_element(Tensor* tensor, u32* indices, double value);

Tensor* add_tensor(Tensor* newTensor, Tensor* tensor1, Tensor* tensor2);
Tensor* sub_tensor(Tensor* newTensor, Tensor* tensor1, Tensor* tensor2);
Tensor* multiply_tensor(Tensor* newTensor, Tensor* tensor1, Tensor* tensor2);
Tensor* divide_tensor(Tensor* newTensor, Tensor* tensor1, Tensor* tensor2);

Tensor* apply_function_tensor(Tensor* tensor, double(*func)(double));

Tensor* reshape_tensor(Tensor* tensor, u32 rank, u32* shape);

Tensor* transpose_tensor(Tensor* tensor);

#endif

/*
*Tensor Basics
Rank (Order): The number of dimensions in a tensor.
Rank 0: Scalar (single value)
Rank 1: Vector (1D array)
Rank 2: Matrix (2D array)
Rank 3+: Higher-order tensor
Shape: The size of each dimension.
Example: A 3x4x2 tensor has shape (3, 4, 2)
Elements: Individual values stored in the tensor, accessed by indices.
Common Tensor Methods
Creation and Initialization
Zeros/Ones: Create tensors filled with 0s or 1s
Random: Generate tensors with random values
From existing data: Convert arrays or lists to tensors
Basic Operations
Arithmetic: Addition, subtraction, multiplication, division
Element-wise operations: Apply functions to each element
Reshaping: Change the tensor's shape while preserving its data
Transposition: Swap dimensions
Advanced Operations
Tensor Contraction:
Summing over indices to reduce tensor rank
Includes matrix multiplication as a special case
Tensor Decomposition:
CP (CANDECOMP/PARAFAC) Decomposition
Tucker Decomposition
Tensor Train Decomposition
Tensor Networks:
Represent high-dimensional tensors as networks of lower-rank tensors
Tensor Products:
Outer product: Combine tensors to create higher-rank tensors
Einstein Summation:
Concise notation for specifying tensor operations
Statistical Methods
Mean, variance, standard deviation along specified axes
Normalization and standardization
Linear Algebra Operations
Eigenvalue decomposition (for matrices)
Singular Value Decomposition (SVD)
QR decomposition
Indexing and Slicing
Accessing and modifying specific elements or subsets of the tensor
Gradient Computation
Automatic differentiation for use in optimization and machine learning
*
*
* */
