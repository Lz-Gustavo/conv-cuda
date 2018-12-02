#include <iostream>
#include <vector>
#include <string>

#include "conv.cuh"

using namespace conv;
using namespace std;


typedef unsigned short* arrayChn;

__global__ 
void ApplyMask(arrayChn old_r, arrayChn old_g, arrayChn old_b, int len, int* kernel, int size_knl, arrayChn red, arrayChn green, arrayChn blue) {

	int n = blockIdx.x;
	if (n < 10) {

		// TODO:
		// Apply kernel mask on the 1-D image representation. Each 'old_channel' are the original image pixels to
		// be modified by the conv tecnique. 'color' arrays store the result of each application.
		// REMEMBER TO JUMP THE FIRST LINES AND COLLUNS AFTER EACH ITERATION (no padding used)
	}
}

void generateImage(string filename, arrayChn red, arrayChn green, arrayChn blue, int offset, int len, int side) {

	unsigned char *output_data = new unsigned char[3 * len];

	int k = 0;
	for (int i = offset; i < offset + len; ++i) {
	
		output_data[k] = (unsigned char) red[i];
		output_data[k+1] = (unsigned char) green[i];
		output_data[k+2] = (unsigned char) blue[i];
		k++;
	}

	if (!tje_encode_to_file(filename.c_str(), side, side, 3, output_data))
		throw(t_ecp);
}

int main(int argc, char **argv) {
	
	if (argc < 2) {
		cout << "Execute with the number of images to be filtered" << endl;
		return 0;
	}

	int NumImg = atoi(argv[1]);
	vector<Image2D*> list_imgs;
	Kernel *k = new Kernel("kernel.txt");

	try {
		for (int id = 1; id <= NumImg; ++id)
			list_imgs.push_back(new Image2D("../../img/simulation/"+to_string(id)+".jpg"));
			
	} catch (exception &e) {
		cout << e.what() << std::endl;
	}

	// all images must have the same resolution
	int size_x = list_imgs[0]->getWidth();
	int size_y = list_imgs[0]->getHeight();
	int len = size_x * size_y;

	unsigned short *red_device, *red_host;
	unsigned short *green_device, *green_host;
	unsigned short *blue_device, *blue_host;

	cudaMalloc((void**) &red_device, (NumImg * len) * sizeof(unsigned short));
	cudaMalloc((void**) &green_device, (NumImg * len) * sizeof(unsigned short));	
	cudaMalloc((void**) &blue_device, (NumImg * len) * sizeof(unsigned short));	

	red_host = new unsigned short[NumImg * len];
	green_host = new unsigned short[NumImg * len];
	blue_host = new unsigned short[NumImg * len];

	int c = 0;
	for (int id = 0; id < NumImg; ++id) {

		unsigned short** red = list_imgs[id]->getRed();
		unsigned short** green = list_imgs[id]->getGreen();
		unsigned short** blue = list_imgs[id]->getBlue();

		for (int i = 0; i < size_x; ++i) {
			for (int j = 0; j < size_y; ++j) {

				red_host[c] = red[j][i];
				green_host[c] = green[j][i];
				blue_host[c] = blue[j][i];
				c++;
			}
		}
	}

	cudaMemcpy(red_device, red_host, (NumImg * len) * sizeof(unsigned short), cudaMemcpyHostToDevice);
	cudaMemcpy(green_device, green_host, (NumImg * len) * sizeof(unsigned short), cudaMemcpyHostToDevice);
	cudaMemcpy(blue_device, blue_host, (NumImg * len) * sizeof(unsigned short), cudaMemcpyHostToDevice);

	//ApplyMask<<<20, 1>>();
	cout << "Acho que ta indo..." << endl;

	cudaMemcpy(red_host, red_device, (NumImg * len) * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cudaMemcpy(green_host, green_device, (NumImg * len) * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cudaMemcpy(blue_host, blue_device, (NumImg * len) * sizeof(unsigned short), cudaMemcpyDeviceToHost);

	for (int i = 0; i < NumImg; i++) {
		generateImage("../../img/simulation/"+to_string(i+1)+"-out.jpg", red_host, green_host, blue_host, i*len, len, size_x);
	}

	cudaFree(red_device);
	cudaFree(green_device);
	cudaFree(blue_device);

	delete[] red_host;
	delete[] green_host;
	delete[] blue_host;
	delete k;

	for (int i = 0; i < list_imgs.size(); i++)
		delete list_imgs[i];

	return 0;
}