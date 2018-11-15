#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define N 5
#define LEN 100
#define THREADS 4

void *BubbleSort(int *vec) {

	int i, j, aux;

	for (i = 0; i < LEN - 1; i++) {
		for (j = i + 1; j < LEN; j++) {
			if (vec[j] < vec[i]) {
				aux = vec[i];
				vec[i] = vec[j];
				vec[j] = aux;
			}
		}
	}
	pthread_exit(NULL);
}

int main() {

	int i, j, c, tmp;
	int **pool;
	pthread_t th[THREADS];

	pool = malloc(N * sizeof(int*));
	for (i = 0; i < N; i++)
		pool[i] = malloc(LEN * sizeof(int));

	srand(time (NULL));

	// Mostra conteudo de todos N arrays pre-ordenacao
	printf("Arrays:\n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < LEN; j++) {
			pool[i][j] = rand() % LEN;
			printf("[%d] = %d\n", j, pool[i][j]);
		}
		printf("\n");
	}

	// TODO: verificar comportamento indeterministico na geracao das threads (me chamaram pra almocar caralho)
	clock_t start = clock();
	for (i = 0; i < N; i++) {

		// Cria o maximo de threads cada THREAD vezes
		if (i % THREADS == 0) {
			c = 0;
			while ((c < THREADS) && (i + c) < N) {
				pthread_create(&th[i], NULL, BubbleSort, (void*) pool[i+c]);
				c++;
			}

			// Espera o termino das threads
			for (c = 0; c < THREADS; c++)
				pthread_join(&th[c], NULL);
		}
	}
	clock_t finish = clock();
	double elapsed = (double)(finish - start) / CLOCKS_PER_SEC;
	
	// Conteudo de todos N arrays apos a ordenacao
	printf("\nArrays Ordenados: (demorou %lf segundos)\n", elapsed);
	for (i = 0; i < N; i++) {
		for (j = 0; j < LEN; j++)
			printf("[%d] = %d\n", j, pool[i][j]);
		printf("\n");
	}

	free(pool);
	return 0;
}