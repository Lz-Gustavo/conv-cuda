#include <stdio.h>
#include <time.h>
#include <glib-2.0/glib.h>

#define N 10
#define LEN 1000

void BubbleSort(int *vec) {

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
}

int main() {

	int i, j;
	int **pool;

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

	GTimer* timer = g_timer_new();
	for (i = 0; i < N; i++)
		BubbleSort(pool[i]);

	g_timer_stop(timer);	
	gulong micro;
	double elapsed = g_timer_elapsed(timer, &micro);

	// Conteudo de todos N arrays apos a ordenacao
	printf("\nArrays Ordenados: \n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < LEN; j++)
			printf("[%d] = %d\n", j, pool[i][j]);
		printf("\n");
	}

	printf("Tempo de Execucao: %lf segundos\n", elapsed);

	free(pool);
	return 0;
}