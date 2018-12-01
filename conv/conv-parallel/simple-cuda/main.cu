#include <iostream>
#include "conv.cuh"

using namespace conv;
using namespace std;

__global__ ApplyMask(unsigned short* image_all, short size_img, int* kernel, short size_knl) {

	// TODO:
	// Apply kernel mask on the 1-D image representation. Remember that 'image_all' is aimmed to contain
	// all pixels from all channels of the extracted img, represented as follow:
	//
	// [ r(0, 0), g(0, 0), b(0, 0), r(0, 1), g(0, 1), b(0, 1), r(0, 2), g(0, 2), b(0, 2), ... ]

}

void generateImage(std::string filename, unsigned short* image_all, int size_x, int size_y) {
	
	int real_size = 3 * (size_x * size_y);
	unsigned char *output_data = new unsigned char[real_size];

	for (int i = 0; i < real_size; ++i) {
			output_data[i] = (unsigned char) image_all[i];
	}

	if (!tje_encode_to_file(filename.c_str(), size_x, size_y, 3, output_data))
		throw(t_ecp);

	delete output_data;
}

int main() {
	
	try {
		
		Kernel *k = new Kernel("kernel.txt");
		k->showMask();

		Image2D *img = new Image2D("img/duck.jpg");
		img->showImage();

		std::cout << std::endl << "======================" << std::endl;


		unsigned short** red = img->getRed(), green = img->getGreen(), blue = img->getBlue();


		ApplyMask<<<20, 1>>();

		generateImage();

		delete k;
		delete img;
	} catch (std::exception &e) {
		std::cout << e.what() << std::endl;
	}
}