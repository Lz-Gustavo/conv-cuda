#include <iostream>
#include <vector>
#include <string>

#include "conv.cuh"

using namespace conv;
using namespace std;

typedef unsigned char* arrayChn;

__global__ 
void ApplyMask(arrayChn old_r, arrayChn old_g, arrayChn old_b, int len, int side, int* kernel, int knl_size, int knl_sum, arrayChn red, arrayChn green, arrayChn blue, int NumImg) {

	int n = blockIdx.x;
	if (n < NumImg) {

		// Apply kernel mask on the 1-D image representation. Each 'old_channel' are the original image pixels to
		// be modified by the conv tecnique. 'color' arrays store the result of each application.
		// REMEMBER TO JUMP THE FIRST LINES AND COLLUNS AFTER EACH ITERATION (no padding used)

		int offset = n * len + side + 1;		// jump the first line and collun
		int fin = offset + len - 1; 			// avoid the last collun

		for(int i = offset; i < fin; ++i) {
			
			short tmp_r = (short) (

				(old_r[i - side - 1] * kernel[0]) + (old_r[i - side] * kernel[1]) + (old_r[i - side + 1] * kernel[2])
				+ (old_r[i - 1] * kernel[3]) + (old_r[i] * kernel[4]) + (old_r[i + 1] * kernel[5])
				+ (old_r[i + side - 1] * kernel[6]) + (old_r[i + side] * kernel[7]) + (old_r[i + side + 1] * kernel[8])

			) / knl_sum;

			if (tmp_r > 255)
				red[i] = (unsigned char) 255;
			else if (tmp_r < 0)
				red[i] = (unsigned char) 0;
			else
				red[i] = (unsigned char) tmp_r;


			short tmp_g = (short) (

				(old_g[i - side - 1] * kernel[0]) + (old_g[i - side] * kernel[1]) + (old_g[i - side + 1] * kernel[2])
				+ (old_g[i - 1] * kernel[3]) + (old_g[i] * kernel[4]) + (old_g[i + 1] * kernel[5])
				+ (old_g[i + side - 1] * kernel[6]) + (old_g[i + side] * kernel[7]) + (old_g[i + side + 1] * kernel[8])
			
			) / knl_sum;

			if (tmp_g > 255)
				green[i] = (unsigned char) 255;
			else if (tmp_g < 0)
				green[i] = (unsigned char) 0;
			else
				green[i] = (unsigned char) tmp_g;


			short tmp_b = (short) (

				(old_b[i - side - 1] * kernel[0]) + (old_b[i - side] * kernel[1]) + (old_b[i - side + 1] * kernel[2])
				+ (old_b[i - 1] * kernel[3]) + (old_b[i] * kernel[4]) + (old_b[i + 1] * kernel[5])
				+ (old_b[i + side - 1] * kernel[6]) + (old_b[i + side] * kernel[7]) + (old_b[i + side + 1] * kernel[8])

			) / knl_sum;

			if (tmp_b > 255)
				blue[i] = (unsigned char) 255;
			else if (tmp_b < 0)
				blue[i] = (unsigned char) 0;
			else
				blue[i] = (unsigned char) tmp_b;
		}
	}
}

void generateImage(string filename, arrayChn red, arrayChn green, arrayChn blue, int offset, int len, int side) {

	unsigned char *output_data = new unsigned char[3 * len];

	int k = 0;
	for (int i = offset; i < offset + len; ++i) {
	
		output_data[k] = (unsigned char) red[i];
		output_data[k+1] = (unsigned char) green[i];
		output_data[k+2] = (unsigned char) blue[i];
		k += 3;
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

	int *knl_host = k->getLinear();
	int knl_size = k->getLinearSize();
	int knl_sum = k->getSum();

	// cout << "kernel: " << endl;
	// for (int i = 0; i < size_knl; ++i) {
	// 	cout << knl_host[i] << " ";
	// }
	// cout << endl;

	try {
		for (int id = 1; id <= NumImg; ++id)
			list_imgs.push_back(new Image2D("../../img/simulation/"+to_string(id)+".jpg"));
	
	} catch (exception &e) {
		cout << e.what() << std::endl;
	}

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	
	// all images must have the same resolution
	int size_x = list_imgs[0]->getWidth();
	int size_y = list_imgs[0]->getHeight();
	int len = size_x * size_y;

	int *knl_device;
	unsigned char *red_device, *red_device_copy, *red_host;
	unsigned char *green_device, *green_device_copy, *green_host;
	unsigned char *blue_device, *blue_device_copy, *blue_host;

	cudaMalloc((void**) &knl_device, knl_size * sizeof(int));
	cudaMalloc((void**) &red_device, (NumImg * len) * sizeof(unsigned char));
	cudaMalloc((void**) &green_device, (NumImg * len) * sizeof(unsigned char));	
	cudaMalloc((void**) &blue_device, (NumImg * len) * sizeof(unsigned char));	
	
	cudaMalloc((void**) &red_device_copy, (NumImg * len) * sizeof(unsigned char));
	cudaMalloc((void**) &green_device_copy, (NumImg * len) * sizeof(unsigned char));	
	cudaMalloc((void**) &blue_device_copy, (NumImg * len) * sizeof(unsigned char));	

	red_host = new unsigned char[NumImg * len];
	green_host = new unsigned char[NumImg * len];
	blue_host = new unsigned char[NumImg * len];

	int c = 0;
	for (int id = 0; id < NumImg; ++id) {

		unsigned char** red = (unsigned char**) list_imgs[id]->getRed();
		unsigned char** green = (unsigned char**) list_imgs[id]->getGreen();
		unsigned char** blue = (unsigned char**) list_imgs[id]->getBlue();

		for (int i = 0; i < size_x; ++i) {
			for (int j = 0; j < size_y; ++j) {

				red_host[c] = red[j][i];
				green_host[c] = green[j][i];
				blue_host[c] = blue[j][i];
				c++;
			}
		}
	}

	// cout << endl << "debug green: " << endl;
	// for (int i = 0; i < NumImg*size_x*size_y; i++) {
	// 	cout << green_host[i] << " ";
	// }
	// cout << endl;

	cudaMemcpy(knl_device, knl_host, (knl_size) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(red_device, red_host, (NumImg * len) * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(green_device, green_host, (NumImg * len) * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(blue_device, blue_host, (NumImg * len) * sizeof(unsigned char), cudaMemcpyHostToDevice);

	cudaMemcpy(red_device_copy, red_host, (NumImg * len) * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(green_device_copy, green_host, (NumImg * len) * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(blue_device_copy, blue_host, (NumImg * len) * sizeof(unsigned char), cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	ApplyMask<<<NumImg, 1>>>(red_device_copy, green_device_copy, blue_device_copy, len, size_x, knl_device, knl_size, knl_sum, red_device, green_device, blue_device, NumImg);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "Elapsed time running " << NumImg << " images: " << milliseconds << "ms" << endl;

	cudaMemcpy(red_host, red_device, (NumImg * len) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(green_host, green_device, (NumImg * len) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(blue_host, blue_device, (NumImg * len) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	for (int i = 0; i < NumImg; i++) {
		generateImage("../../img/simulation/"+to_string(i+1)+"-out.jpg", red_host, green_host, blue_host, i*len, len, size_x);
	}

	cudaFree(knl_device);
	cudaFree(red_device);
	cudaFree(green_device);
	cudaFree(blue_device);

	delete[] knl_host;
	delete[] red_host;
	delete[] green_host;
	delete[] blue_host;
	delete k;

	for (int i = 0; i < list_imgs.size(); i++)
		delete list_imgs[i];

	return 0;
}