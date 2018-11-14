#include <iostream>
#include "conv.h"

using namespace conv;
using namespace std;

int main() {

	Kernel *k = new Kernel("./kernel.txt");
	k->showMask();

	Image2D *img = new Image2D("./duck.jpg");
	img->showImage();

	std::cout << std::endl << "======================" << std::endl;
	
	img->ApplyMask(k->getMask());
	img->showImage();
	
	delete k;
	delete img;
}