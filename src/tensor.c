#include "../libs/tensor.h"
#include "../libs/util.h"
#include <_strings.h>
#include <stdlib.h>

Tensor* create_tensor(u32 rank, u32* shape) {
	if (rank < 1) return NULL;

	Tensor* tensor = malloc(sizeof(Tensor));
	if (tensor == NULL) return NULL;
	tensor->rank = rank;

	u32* dim = malloc(sizeof(u32) * rank);
	if(dim == NULL) return NULL;

	u64 size = 1;
	for (u32 i = 0; i < rank; i++) {
		size *= shape[i];
		dim[i] = shape[i];
	}
	tensor->size = size;

	tensor->shape = dim;
	tensor->data =  malloc(sizeof(double) * size);
	if (tensor->data == NULL) return NULL;

	return tensor;
}

Tensor* create_tensor_zeros(u32 rank, u32* shape) {
	if (rank < 1) return NULL;

	Tensor* tensor = malloc(sizeof(Tensor));
	if (tensor == NULL) return NULL;
	tensor->rank = rank;

	u32* dim = malloc(sizeof(u32) * rank);
	if (dim == NULL) return NULL;

	u64 size = 1;
	for (u32 i = 0; i < rank; i++) {
		size *= shape[i];
		dim[i] = shape[i];
	}
	tensor->size = size;

	tensor->shape = dim;
	tensor->data = calloc(sizeof(double), size);
	if (tensor->data == NULL) return NULL;

	return tensor;
}

Tensor* create_tensor_random(u32 rank, u32* shape) {
	if (rank < 1) return NULL;

	Tensor* tensor = create_tensor(rank, shape);

	for (u32 i = 0; i < tensor->size; i++) {
		tensor->data[i] = normalized_random();
	}

	return tensor;
}

// Tensor* create_tensor_array(double* source, u32 rank, u32* shape);

void free_tensor(Tensor* tensor) {
	if (tensor == NULL) return;

	if (tensor->data) free(tensor->data);
	if (tensor->shape) free(tensor->shape);

	free(tensor);
}

float get_element(Tensor* tensor, u32* indices) {
	u64 index = 0;
	u32 multiplier = 1;

	for (u32 i = tensor->rank - 1; i >= 0; i--) {
		index += indices[i] * multiplier;
		multiplier *= tensor->shape[i];
	}

	return tensor->data[index];
}

void set_element(Tensor* tensor, u32* indices, double value) {
	u64 index = 0;
	u32 multiplier = 1;

	for (u32 i = tensor->rank - 1; i >= 0; i--) {
		index += indices[i] * multiplier;
		multiplier *= tensor->shape[i];
	}

	tensor->data[index] = value;
}

Tensor* add_tensor(Tensor* newTensor, Tensor* tensor1, Tensor* tensor2) {
	if (newTensor == NULL || tensor1 == NULL || tensor2 == NULL) return NULL;
	if (tensor1->rank != tensor2->rank) return NULL;
	if (tensor1->size != tensor2->size) return NULL;

	for (u64 i = 0; i < newTensor->size; i++) {
		newTensor->data[i] = tensor1->data[i] + tensor2->data[i];
	}

	return newTensor;
}

Tensor* sub_tensor(Tensor* newTensor, Tensor* tensor1, Tensor* tensor2) {
	if (newTensor == NULL || tensor1 == NULL || tensor2 == NULL) return NULL;
	if (tensor1->rank != tensor2->rank) return NULL;
	if (tensor1->size != tensor2->size) return NULL;

	for (u64 i = 0; i < newTensor->size; i++) {
		newTensor->data[i] = tensor1->data[i] - tensor2->data[i];
	}

	return newTensor;
}

Tensor* multiply_tensor(Tensor* newTensor, Tensor* tensor1, Tensor* tensor2) {
	if (newTensor == NULL || tensor1 == NULL || tensor2 == NULL) return NULL;
	if (tensor1->rank != tensor2->rank) return NULL;
	if (tensor1->size != tensor2->size) return NULL;

	for (u64 i = 0; i < newTensor->size; i++) {
		newTensor->data[i] = tensor1->data[i] * tensor2->data[i];
	}

	return newTensor;
}

Tensor* divide_tensor(Tensor* newTensor, Tensor* tensor1, Tensor* tensor2) {
	if (newTensor == NULL || tensor1 == NULL || tensor2 == NULL) return NULL;
	if (tensor1->rank != tensor2->rank) return NULL;
	if (tensor1->size != tensor2->size) return NULL;

	for (u64 i = 0; i < newTensor->size; i++) {
		newTensor->data[i] = tensor1->data[i] / tensor2->data[i];
	}

	return newTensor;
}

Tensor* apply_function_tensor(Tensor* tensor, double(*func)(double)) {
	if (tensor == NULL) return NULL;
	if (func == NULL) return NULL;

	for (u64 i = 0; i < tensor->size; i++) {
		tensor->data[i] = func(tensor->data[i]);
	}

	return tensor;
}

Tensor* reshape_tensor(Tensor* tensor, u32 rank, u32* shape) {
	if (tensor == NULL) return NULL;
	if (shape == NULL) return NULL;
	if (rank < 1) return NULL;

	u64 size = 1;
	for (u32 i = 0; i < rank; i++) {
		size *= shape[i];
	}

	if (size != tensor->size) return NULL;

	free(tensor->shape);

	u32* newShape = malloc(sizeof(u32) * rank);
	for (u32 i = 0; i < rank; i++) {
		newShape[i] = shape[i];
	}

	tensor->rank = rank;
	tensor->shape = newShape;

	return tensor;
}

// Tensor* transpose_tensor(Tensor* tensor);
