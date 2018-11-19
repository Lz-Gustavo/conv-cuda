#include <stdlib.h>
#include <stdio.h>

#define ARR_SIZE 10

__global__ void add(int *a, int *b, int *c) {

	int i = blockIdx.x;

	if (i < ARR_SIZE)
		c[i] = a[i] + b[i];
}

int main() {
	
	int i;

	int h_A[ARR_SIZE], h_B[ARR_SIZE], h_C[ARR_SIZE];
	int *d_A, *d_B, *d_C;

	// Popula os vetores a serem somados
	for (i = 0; i < ARR_SIZE; i++) {
		h_A[i] = i;
		h_B[i] = i + 1;	
	}

	// Aloca-se memória para os três no dispositivo (GPU)
	cudaMalloc((void**) &d_A, ARR_SIZE * sizeof(int));
	cudaMalloc((void**) &d_B, ARR_SIZE * sizeof(int));
	cudaMalloc((void**) &d_C, ARR_SIZE * sizeof(int));

	// Copia o conteudo DRAM -> VRAM
	cudaMemcpy(d_A, h_A, ARR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, ARR_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	// Despacha ARR_SIZE blocos de execucao paralela na GPU
	add<<<ARR_SIZE, 1>>> (d_A, d_B, d_C);

	// Copia o conteudo VRAM -> DRAM
	cudaMemcpy(h_C, d_C, ARR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	for (i = 0; i < ARR_SIZE; i++) {
		printf ("[%d] -> %d + %d = %d\n", i, h_A[i], h_B[i], h_C[i]);
	}

	// Desaloca memoria da GPU
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}