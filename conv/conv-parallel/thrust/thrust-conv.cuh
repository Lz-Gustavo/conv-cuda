#ifndef CONV_H
#define CONV_H

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "exceptions.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../stb_image.h"
#define USE_MGL_NAMESPACE

#define TJE_IMPLEMENTATION
#include "../../tiny_jpeg.h"

namespace conv {

	class Kernel;

	// template<typename T>
	// struct conv_functor {

	// 	Kernel _k;

	// 	conv_functor(Kernel k) : _k(k) {}

	// 	typedef T pixel;

	// 	__device__ T operator()(const T &x) const {

			
	// 	}
	// };

	class Image2D {
	private:
		
		unsigned char* image;
		thrust::device_vector<unsigned char> image_th;
		int size_y;
		int size_x;
		int nrChannels;

		thrust::device_vector<unsigned short> image_copy_r;
		thrust::device_vector<unsigned short> image_copy_g;
		thrust::device_vector<unsigned short> image_copy_b;

	public:
		Image2D(std::string filename) {

			image = stbi_load(filename.c_str(), &size_x, &size_y, &nrChannels, 3);			
			if (!image) {
				throw (d_ecp);
				return;
			}
			
			for (int i = 0; i < size_y; ++i) {				
				for (int j = 0; j < size_x; ++j) {
					
					unsigned char *aux = getPixel(i, j);

					image_copy_r.push_back(aux[0]);
					image_copy_g.push_back(aux[1]);
					image_copy_b.push_back(aux[2]);
				}
			}
		}
		~Image2D() {
			stbi_image_free(image);

			image_copy_r.clear();
			image_copy_r.shrink_to_fit();

			image_copy_g.clear();
			image_copy_g.shrink_to_fit();
			
			image_copy_b.clear();
			image_copy_b.shrink_to_fit();
		}

		int getWidth() {
			return size_x;
		}
		int getHeight() {
			return size_y;
		}

		unsigned char* getPixel(int i, int j) {
			return image + (i + size_y * j) * nrChannels;
		}

		unsigned short getPixel(int i, int j, int channel) {
			// Channels:
			// 0- Red
			// 1- Green
			// 2- Blue

			unsigned char *pixelOffset = image + (i + size_y * j) * nrChannels;
			return pixelOffset[channel];
		}

		void ApplyMask(std::vector<std::vector<int>*> *kernel) {

			// TODO:
			// implement a convolutional filter functor like 
			// https://stackoverflow.com/questions/33923645/thrustdevice-vector-use-thrustreplace-or-thrusttransform-with-custom-funct
			// and apply it to all three image channels using thrust::transform
		
		}

		void showImage() {

			std::cout << "Red Pixel map: " << std::endl;
			for (int i = 0; i < image_copy_r.size(); ++i) {
				std::cout << image_copy_r[i] << " ";
				
				if ((i+1) % size_x == 0)
					std::cout << std::endl;
			}

			std::cout << std::endl << "Green Pixel map: " << std::endl;
			for (int i = 0; i < image_copy_g.size(); ++i) {
				std::cout << image_copy_g[i] << " ";
				
				if ((i+1) % size_x == 0)
					std::cout << std::endl;
			}

			std::cout << std::endl << "Blue Pixel map: " << std::endl;
			for (int i = 0; i < image_copy_b.size(); ++i) {
				std::cout << image_copy_b[i] << " ";
				
				if ((i+1) % size_x == 0)
					std::cout << std::endl;
			}
		}

		void generateImage(std::string filename) {

			// TODO:
			// copy all pixels values from all three device vectors and generate a jpeg image,
			// simalar as done in conv.h
		}
	};

	class Kernel {
	private:
		std::vector<std::vector<int>*> *mask;
		int size_x;
		int size_y;

	public:
		Kernel(std::string filename) {

			std::fstream file;	
			std::string line;
			std::stringstream byte, aux_space;
			
			file.open(filename, std::ios::in);
			getline(file, line);
			byte << line;
			byte >> size_x;
			byte >> size_y;
			byte.clear();

			mask = new std::vector<std::vector<int>*>();
			while (getline(file, line)) {
				
				byte << line;
				int aux_number;
				std::vector<int> *row = new std::vector<int>;
				for (int i = 0; i < size_x; ++i) {
					byte >> aux_number;
					row->push_back(aux_number);
				}
				mask->push_back(row);
				byte.clear();
			}
			file.close();
		}
		~Kernel() {
			
			for (int i = 0; i < mask->size(); ++i) {
				delete mask->at(i);
			}
			delete mask;
		}

		std::vector<std::vector<int>*>* getMask() {
			return mask;
		}

		void showMask() {

			std::cout << "Kernel mask: " << std::endl;
			for (int i = 0; i < mask->size(); ++i) {
				for (int j = 0; j < mask->at(i)->size(); ++j) {

					std::cout << mask->at(i)->at(j) << " ";
				}
				std::cout << std::endl;
			}
		}
	};
}

#endif