#include <iostream>
#include <vector>

#include "../conv.h"

using namespace std;

int main() {

	conv::Kernel *k = new conv::Kernel("input2.txt");

	std::vector<std::vector<int>*> *m = k->getMask();


	for (;;) {

		int n;

		std::cout << "Original image: " << std::endl;
		k->showMask();
		cout << endl;

		std::cout << "Divide in how much tiles? " << std::endl;
		std::cin >> n;
		if (n == -1)
			break;
		else if (n % 2 != 0) {
			std::cout << "Insert an even number of tiles." << std::endl;
			continue;
		}

		// initialize empty values on 'N' dimensions of out (represents each desired tile)
		//vector<vector<vector<int>>> out;
		//out.resize(n);
	
		int size = m->size();
		int stride = size/(n/2);
		int out[n][stride][stride];

		for (int i = 0; i < stride; i++) {
			for (int j = 0; j < stride; j++) {
				
				int a, b;
				for (int k = 0; k < n; k++) {

					if (k % 2 != 0)
						b = stride;
					else
						b = 0;
					
					if (k >= n/2)
						a = stride;
					else
						a = 0;

					//cout << "a = " << a << " b = " << b << endl;

					out[k][i][j] = m->at(i + a)->at(j + b);
				}
			}
		}

		//print tiles
		for (int k = 0; k < n; k++) {

			cout << "TILE " << k << endl;
			for (int i = 0; i < stride; i++) {
				for (int j = 0; j < stride; j++) {

					cout << out[k][i][j] << " ";
				}
				cout << endl;
			}
			cout << endl << endl;
		}
	}

	delete k;
	for (int i = 0; i < m->size(); i++) {
		delete[] m->at(i);
	}
	delete[] m;

	return 0;
}