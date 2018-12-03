#ifndef CONV_H
#define CONV_H

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

	class Image2D {
	private:
		unsigned char* image;
		int size_y;
		int size_x;
		int nrChannels;
		unsigned short** image_copy_r;
		unsigned short** image_copy_g;
		unsigned short** image_copy_b;

	public:
		Image2D(std::string filename) {

			image = stbi_load(filename.c_str(), &size_x, &size_y, &nrChannels, 3);			
			if (!image) {
				throw (d_ecp);
				return;
			}

			image_copy_r = new unsigned short*[size_y];
			image_copy_g = new unsigned short*[size_y];
			image_copy_b = new unsigned short*[size_y];
			
			for (int i = 0; i < size_y; ++i) {

				image_copy_r[i] = new unsigned short[size_x];
				image_copy_g[i] = new unsigned short[size_x];
				image_copy_b[i] = new unsigned short[size_x];
				
				for (int j = 0; j < size_x; ++j) {
					
					unsigned char *aux = getPixel(i, j);

					image_copy_r[i][j] = aux[0];
					image_copy_g[i][j] = aux[1];
					image_copy_b[i][j] = aux[2];
				}
			}
		}
		~Image2D() {
			stbi_image_free(image);

			for (int i = 0; i < size_x; ++i) {
				delete[] image_copy_r[i];
				delete[] image_copy_g[i];
				delete[] image_copy_b[i];
			}
			delete[] image_copy_r;
			delete[] image_copy_g;
			delete[] image_copy_b;
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

		void showImage() {

			std::cout << "Red Pixel map: " << std::endl;
			for (int i = 0; i < size_y; ++i) {
				for (int j = 0; j < size_x; ++j) {

					std::cout << image_copy_r[i][j] << " ";
				}
				std::cout << std::endl;
			}

			std::cout << std::endl << "Green Pixel map: " << std::endl;
			for (int i = 0; i < size_y; ++i) {
				for (int j = 0; j < size_x; ++j) {

					std::cout << image_copy_g[i][j] << " ";
				}
				std::cout << std::endl;
			}

			std::cout << std::endl << "Blue Pixel map: " << std::endl;
			for (int i = 0; i < size_y; ++i) {
				for (int j = 0; j < size_x; ++j) {

					std::cout << image_copy_b[i][j] << " ";
				}
				std::cout << std::endl;
			}
		}

		unsigned short** getRed() {
			return image_copy_r;
		}

		unsigned short** getGreen() {
			return image_copy_g;
		}

		unsigned short** getBlue() {
			return image_copy_b;
		}

		void setRed(unsigned short** r) {
			image_copy_r = r;
		}

		void setGreen(unsigned short** g) {
			image_copy_g = g;
		}

		void setBlue(unsigned short** b) {
			image_copy_b = b;
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

		int *getLinear() {
			
			int *line = new int[size_x * size_y];

			int k = 0;
			for (int i = 0; i < size_x; ++i) {
				for (int j = 0; j < size_y; ++j) {
					line[k] = mask->at(j)->at(i);
					k++;
				}
			}

			return line;
		}

		int getLinearSize() {
			return size_x*size_y;
		}

		int getSum() {

			int mask_summ;

			for (int i = 0; i < mask->size(); ++i) {
				for (int j = 0; j < mask->at(i)->size(); ++j) {
					mask_summ += mask->at(i)->at(j);
				}
			}

			return mask_summ;
		}
	};
}

#endif