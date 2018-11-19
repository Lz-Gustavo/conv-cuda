#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <glib-2.0/glib.h>

#define N 10
#define LEN 1000
#define THREADS 4

__global__ void BubbleSort(int *vec) {

	int n = blockIdx.x;

	if (n < N) {

		int i, j, aux;
		int ini = n * LEN;
		int fin = ini + LEN;

		for (i = ini; i < fin - 1; i++) {
			for (j = i + 1; j < fin; j++) {

				if (vec[j] < vec[i]) {
					aux = vec[i];
					vec[i] = vec[j];
					vec[j] = aux;
				}
			}
		}
	}
}

int main() {

	int *arrays_cpu;
	int *arrays_device;
	
	// Melhor performance ao utilizar uma representacao de 1 dimensão na GPU
	arrays_cpu = malloc(N * LEN * sizeof(int));
	cudaMalloc((void**) &arrays_device, N * LEN * sizeof(int));

	srand(time (NULL));

	// Mostra conteudo de todos N arrays pre-ordenacao
	printf("Arrays:\n");
	for (int i = 0; i < (N * LEN); i++) {
		arrays_cpu[i] = rand() % LEN;
		printf("[%d] = %d\n", i, arrays_cpu[i]);
		if (i % N == 0)
			printf("\n");
	}

	// Copia conteudo DRAM -> VRAM
	cudaMemcpy(arrays_device, arrays_cpu, N * LEN * sizeof(int), cudaMemcpyHostToDevice);

	// Inicia contagem tempo de execucao
	GTimer* timer = g_timer_new();

	// Despacha N blocos de execucao paralela na GPU
	BubbleSort<<<N, 1>>> (arrays_device);

	// Copia conteudo VRAM -> DRAM
	cudaMemcpy(arrays_cpu, arrays_device, N * LEN * sizeof(int), cudaMemcpyDeviceToHost);

	g_timer_stop(timer);	
	gulong micro;
	double elapsed = g_timer_elapsed(timer, &micro);
		
	// Conteudo de todos N arrays apos a ordenacao
	for (int i = 0; i < (N * LEN); i++) {
		
		printf("[%d] = %d\n", i, arrays_cpu[i]);
		if (i % N == 0)
			printf("\n");
	}

	printf("Tempo de Execucao: %lf segundos\n", elapsed);

	cudaFree(arrays_device);
	free(arrays_cpu);
	return 0;
}