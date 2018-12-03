#include <iostream>
#include <vector>
#include <glib-2.0/glib.h>
#include "conv.h"

using namespace conv;
using namespace std;

int main(int argc, char **argv) {
	

	if (argc < 2) {
		cout << "Execute with the number of images to be filtered" << endl;
		return 0;
	}
	
	vector<Image2D*> list_imgs;

	Kernel *k = new Kernel("kernel.txt");
	int NumImg = atoi(argv[1]);

	try {
		for (int id = 1; id <= NumImg; ++id)
			list_imgs.push_back(new Image2D("img/google/animal/"+to_string(id)+".jpg"));
	
		for (int i = 0; i < NumImg; ++i)
			list_imgs[i]->ApplyMask(k->getMask());
		
		GTimer* timer = g_timer_new();
		for (int i = 0; i < NumImg; ++i)
			list_imgs[i]->generateImage("img/google/animal/"+to_string(i+1)+"-out.jpg");
		
		g_timer_stop(timer);	
		gulong micro;
		double elapsed = g_timer_elapsed(timer, &micro);

		cout << "Tempo de Execucao: " << elapsed << "seg" << endl;

		delete k;
		for (int i = 0; i < NumImg; ++i)
			delete list_imgs[i];

	} catch (exception &e) {
		cout << e.what() << std::endl;
	}
}