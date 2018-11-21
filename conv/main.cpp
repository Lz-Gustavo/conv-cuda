#include <iostream>
#include "conv.h"

using namespace conv;
using namespace std;

int main() {
	
	try {
		
		Kernel *k = new Kernel("kernel.txt");
		k->showMask();

		Image2D *img = new Image2D("img/duck.jpg");
		img->showImage();

		std::cout << std::endl << "======================" << std::endl;
		
		img->ApplyMask(k->getMask());
		img->showImage();
		
		img->generateImage("img/duck-test.jpg");

		delete k;
		delete img;
	} catch (std::exception &e) {
		std::cout << e.what() << std::endl;
	}
}