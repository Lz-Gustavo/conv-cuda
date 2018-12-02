#include <iostream>
#include "conv.cuh"

using namespace conv;
using namespace std;

// typedef struct {

// 	int size_x;
// 	int size_y;

// 	unsigned short *red;
// 	unsigned short *green;
// 	unsigned short *blue;

// } Image;

__global__ void ApplyMask(unsigned short* image_all, short size_img, int* kernel, short size_knl) {

	// TODO:
	// Apply kernel mask on the 1-D image representation. Remember that 'image_all' is aimmed to contain
	// all pixels from all channels of the extracted img, represented as follow:
	//
	// [ r(0, 0), g(0, 0), b(0, 0), r(0, 1), g(0, 1), b(0, 1), r(0, 2), g(0, 2), b(0, 2), ... ]

}

void generateImage(std::string filename, Image2D *img) {
	
	int size_x = img->getWidth();
	int size_y = img->getHeight();

	int real_size = 3 * (size_x * size_y);
	unsigned char *output_data = new unsigned char[real_size];
	unsigned short **red = img->getRed(), **green = img->getGreen(), **blue = img->getBlue();

	int k = 0;
	for (int i = 0; i < size_x; ++i) {
		
		for (int j = 0; j < size_y; ++j) {

			output_data[k] = (unsigned char) red[j][i];
			output_data[k+1] = (unsigned char) green[j][i];
			output_data[k+2] = (unsigned char) blue[j][i];
			k += 3;
		}

		delete[] red[i];
		delete[] green[i];
		delete[] blue[i];
	}

	if (!tje_encode_to_file(filename.c_str(), size_x, size_y, 3, output_data))
		throw(t_ecp);

	delete output_data;
	delete[] red;
	delete[] green;
	delete[] blue;
}

int main() {
	
	try {
		
		Kernel *k = new Kernel("kernel.txt");
		Image2D *img = new Image2D("../../img/duck.jpg");

		int size_x = img->getWidth();
		int size_y = img->getHeight();
		
		unsigned short** red = img->getRed();
		unsigned short** green = img->getGreen();
		unsigned short** blue = img->getBlue();

		unsigned short *red_device, *red_host;
		unsigned short *green_device, *green_host;
		unsigned short *blue_device, *blue_host;

		cudaMalloc((void**) &red_device, (size_x * size_y) * sizeof(unsigned short));
		cudaMalloc((void**) &green_device, (size_x * size_y) * sizeof(unsigned short));	
		cudaMalloc((void**) &blue_device, (size_x * size_y) * sizeof(unsigned short));	

		red_host = new unsigned short[size_x * size_y];
		green_host = new unsigned short[size_x * size_y];
		blue_host = new unsigned short[size_x * size_y];

		int c = 0;
		for (int i = 0; i < size_x; ++i) {
			for (int j = 0; j < size_y; ++j) {

				red_host[c] = red[i][j];
				green_host[c] = green[i][j];
				blue_host[c] = blue[i][j];
				c++;
			}
		}

		cudaMemcpy(red_device, red_host, (size_x * size_y) * sizeof(unsigned short), cudaMemcpyHostToDevice);
		cudaMemcpy(green_device, green_host, (size_x * size_y) * sizeof(unsigned short), cudaMemcpyHostToDevice);
		cudaMemcpy(blue_device, blue_host, (size_x * size_y) * sizeof(unsigned short), cudaMemcpyHostToDevice);

		//ApplyMask<<<20, 1>>();

		cout << "Acho que ta indo..." << endl;

		//generateImage();

		cudaFree(red_device);
		cudaFree(green_device);
		cudaFree(blue_device);

		// for (int i = 0; i < size_x; ++i) {
		// 	delete[] red[i];
		// 	delete[] green[i];
		// 	delete[] blue[i];
		// }

		//delete[] red;
		delete[] red_host;
		//delete[] green;
		delete[] green_host;
		//delete[] blue;
		delete[] blue_host;

		delete k;
		delete img;

	} catch (std::exception &e) {
		std::cout << e.what() << std::endl;
	}
}