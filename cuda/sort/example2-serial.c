#include <stdio.h>
#include <time.h>

#define N 100

void BubbleSort(int *vec) {

	int i, j, aux;

	for (i = 0; i < N - 1; i++) {
		for (j = i + 1; j < N; j++) {
			if (vec[j] < vec[i]) {
				aux = vec[i];
				vec[i] = vec[j];
				vec[j] = aux;
			}
		}
	}
}

int main() {

	int i;
	int array[N];

	srand(time (NULL));

	printf("Array:\n");
	for (i = 0; i < N; i++) {
		array[i] = rand() % N;
		printf("[%d] = %d\n", i, array[i]);
	}

	clock_t start = clock();
	BubbleSort(array);
	clock_t finish = clock();
	
	double elapsed = (double)(finish - start) / CLOCKS_PER_SEC;

	printf("\nArray Ordenado: (demorou %lf segundos)\n", elapsed);
	for (i = 0; i < N; i++)
		printf("[%d] = %d\n", i, array[i]);
	printf("\n");

	return 0;
}